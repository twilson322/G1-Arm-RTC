#include "g1rtc/observer.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "g1rtc/controllers.hpp"  // lowPassGain

namespace g1rtc {

MomentumObserver::MomentumObserver(const rbd::Model& m, Params p)
    : model_(&m), p_(std::move(p)), kin_(m), work_(m), kin_dual_(m), work_dual_(m) {
  if (p_.gain_per_s.size() != m.nv) throw std::invalid_argument("observer gain needs one entry per DOF");
  momentum_integral_ = residual_ = VecN::Zero(m.nv);
}

void MomentumObserver::reset(const VecN& q, const VecN& qd) {
  rbd::forwardKinematics(*model_, q, kin_);
  rbd::crba(*model_, kin_, work_, M_);
  momentum_integral_ = M_ * qd;  // p(0)
  residual_.setZero(model_->nv);
  filters_initialized_ = false;
}

const VecN& MomentumObserver::update(const VecN& q, const VecN& qd, const VecN& mean_applied_torque, double dt_s) {
  const VecN beta = rbd::observerBias(*model_, q, qd, kin_, work_, kin_dual_, work_dual_);
  rbd::crba(*model_, kin_, work_, M_);  // kin_ was just evaluated at q by observerBias
  VecN tau = mean_applied_torque;
  if (p_.model_joint_friction)
    for (int i = 0; i < model_->nv; ++i) tau[i] -= p_.coulomb_friction_nm * std::tanh(qd[i] / p_.friction_smoothing_rad_s);
  if (p_.drive_velocity_filter_hz > 0.0) {
    if (!filters_initialized_) {
      tau_filtered_ = tau;
      beta_filtered_ = beta_previous_ = beta;
      filters_initialized_ = true;
    }
    const double a = lowPassGain(p_.drive_velocity_filter_hz, dt_s);
    tau_filtered_ += a * (tau - tau_filtered_);
    // trapezoidal beta, to match tau, which is an average over the interval
    beta_filtered_ += a * (0.5 * (beta + beta_previous_) - beta_filtered_);
    beta_previous_ = beta;
    momentum_integral_ += (tau_filtered_ - beta_filtered_ + residual_) * dt_s;
  } else {
    momentum_integral_ += (tau - beta + residual_) * dt_s;
  }
  residual_ = p_.gain_per_s.cwiseProduct(M_ * qd - momentum_integral_);
  return residual_;
}

bool CollisionDetector::update(const VecN& residual) {
  const bool above = (residual.cwiseAbs().array() > p_.threshold_nm.array()).any();
  if (!above) {
    consecutive_ = 0;
    return false;
  }
  if (tripped_) return false;
  if (++consecutive_ >= p_.persistence_ticks) {
    tripped_ = true;
    return true;
  }
  return false;
}

void CollisionDetector::rearm() {
  consecutive_ = 0;
  tripped_ = false;
}

int isolateContactBody(const rbd::Model& m, const rbd::Kinematics<double>& k, const VecN& residual, const VecN& joint_scale,
                       const std::vector<int>& candidates, double tie_tolerance) {
  // Runs on the control thread, so no throwing: ControlStack checks the count at construction.
  const std::size_t count = std::min(candidates.size(), kMaxContactCandidates);
  const VecN r = residual.cwiseQuotient(joint_scale);
  const double r_norm = r.norm();
  if (r_norm < 1e-9 || candidates.empty()) return -1;
  Mat6N J;
  std::array<double, kMaxContactCandidates> fit_error{};
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < count; ++i) {
    const int body = candidates[i];
    rbd::pointJacobian(m, k, body, k.com[static_cast<std::size_t>(body)], J);
    Mat3N Jv = J.topRows<3>();
    for (int j = 0; j < m.nv; ++j) Jv.col(j) /= joint_scale[j];
    const Vec3 force = (Jv * Jv.transpose() + 1e-9 * Mat3::Identity()).ldlt().solve(Jv * r);
    const VecN unexplained = r - Jv.transpose() * force;
    fit_error[i] = unexplained.norm() / r_norm;
    best = std::min(best, fit_error[i]);
  }
  for (std::size_t i = 0; i < count; ++i)
    if (fit_error[i] <= best + tie_tolerance) return candidates[i];
  return -1;
}

Vec3 ToolWrenchEstimator::contactForceWorld(const Vec6& reading, const Mat3& R) const {
  // The sensor reports the force the wrist exerts on the tool. Static balance of the tool:
  // f_wrist + f_contact + m g = 0.
  const Vec3 wrist_on_tool = R * (reading.head<3>() - force_offset_sensor_);
  return -wrist_on_tool - tool_mass_kg_ * gravity_;
}

void ToolWrenchEstimator::accumulateTare(const Vec6& reading, const Mat3& R) {
  // in free space the reading should be R^T (-m g); whatever else there is, is offset
  offset_sum_ += reading.head<3>() - R.transpose() * (-tool_mass_kg_ * gravity_);
  ++offset_samples_;
  force_offset_sensor_ = offset_sum_ / offset_samples_;
}

void ToolWrenchEstimator::clearTare() {
  offset_sum_.setZero();
  offset_samples_ = 0;
}

PayloadEstimator::PayloadEstimator(Params p) : p_(p) { reset(); }

void PayloadEstimator::reset() {
  theta_.setZero();
  P_ = p_.initial_covariance * Eigen::Matrix4d::Identity();
}

Vec3 PayloadEstimator::comToolFrame() const {
  constexpr double kMinMassForComKg = 0.05;  // below this m c / m is noise over noise
  if (theta_[0] < kMinMassForComKg) return Vec3::Zero();
  Vec3 com = theta_.tail<3>() / theta_[0];
  if (com.norm() > p_.max_com_offset_m) com *= p_.max_com_offset_m / com.norm();
  return com;
}

PayloadEstimator::Regressor PayloadEstimator::regressor(const rbd::DynamicsCache& dyn, const Vec3& gravity) {
  const int n = static_cast<int>(dyn.J.cols());
  Regressor Y(n, 4);
  const Mat3 gravity_cross = skew(gravity);
  for (int i = 0; i < n; ++i) {
    Y(i, 0) = dyn.J.block<3, 1>(0, i).dot(gravity);
    Y.block<1, 3>(i, 1) = -(dyn.J.block<3, 1>(3, i).transpose() * gravity_cross * dyn.tool_rotation);
  }
  return Y;
}

void PayloadEstimator::update(const rbd::DynamicsCache& dyn, const VecN& residual, const Vec3& gravity,
                              const Eigen::Vector4d& already_modelled) {
  const Regressor Y = regressor(dyn, gravity);
  const VecN target = residual + Y * already_modelled;
  // one scalar RLS update per joint: no matrix inverse, and a joint can be skipped cheaply
  for (int i = 0; i < static_cast<int>(residual.size()); ++i) {
    Eigen::Vector4d y = Y.row(i).transpose();
    if (!p_.estimate_com) y.tail<3>().setZero();
    const double denom = p_.forgetting + y.dot(P_ * y);
    const Eigen::Vector4d gain = P_ * y / denom;
    theta_ += gain * (target[i] - y.dot(theta_));
    P_ = (P_ - gain * y.transpose() * P_) / p_.forgetting;
  }
  // bound the covariance so it cannot wind up while the motion does not excite some direction
  const double max_trace = 4 * p_.initial_covariance;
  if (P_.trace() > max_trace) P_ *= max_trace / P_.trace();
}

}  // namespace g1rtc
