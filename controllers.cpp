#include "g1rtc/controllers.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace g1rtc {

Reference Reference::hold(const VecN& q) {
  Reference r;
  r.q = q;
  r.qd = VecN::Zero(q.size());
  r.qdd = VecN::Zero(q.size());
  return r;
}

int saturate(const rbd::Model& m, VecN& tau) {
  int clipped = 0;
  for (int i = 0; i < m.nv; ++i) {
    const double limit = m.tau_max[i];
    if (std::abs(tau[i]) > limit) {
      tau[i] = std::clamp(tau[i], -limit, limit);
      ++clipped;
    }
  }
  return clipped;
}

JointPdGravity JointPdGravity::defaultsG1() {
  // Hand-tuned per joint group to roughly 3-6 Hz at the home posture (stiffness over the
  // diagonal of M), with damping near 0.7. The wrist gains are low because their 5 N m limit and
  // tiny inertia make them the first to chatter on the filtered velocity.
  // order: waist yaw/roll/pitch, shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
  VecN kp(10), kd(10);
  kp << 400, 400, 400, 120, 120, 80, 80, 25, 12, 12;
  kd << 12, 12, 12, 4, 4, 2.5, 2.5, 0.6, 0.3, 0.3;
  return {kp, kd};
}

VecN JointPdGravity::compute(const ControlContext& c, const Reference& r) {
  return VecN(kp_.cwiseProduct(r.q - c.q) + kd_.cwiseProduct(r.qd - c.qd) + c.dyn.g);
}

ComputedTorque::ComputedTorque(double bandwidth_hz, double damping_ratio) {
  const double wn = 2.0 * std::numbers::pi * bandwidth_hz;
  kp_per_s2_ = wn * wn;
  kd_per_s_ = 2.0 * damping_ratio * wn;
}

VecN ComputedTorque::compute(const ControlContext& c, const Reference& r) {
  const VecN commanded_acceleration = r.qdd + kp_per_s2_ * (r.q - c.q) + kd_per_s_ * (r.qd - c.qd);
  return VecN(c.dyn.M * commanded_acceleration + c.dyn.h);
}

void OscTerms::compute(const ControlContext& c, const Reference& r, const TaskGains& g) {
  const auto& d = c.dyn;
  const MatN6 Minv_JT = d.M_cholesky.solve(d.J.transpose());
  const Mat6 Lambda_inverse = d.J * Minv_JT;
  Lambda = (Lambda_inverse + g.lambda_regularization * Mat6::Identity()).ldlt().solve(Mat6::Identity());
  Jbar_transpose = (Minv_JT * Lambda).transpose();
  tool_twist = d.J * c.qd;
  // posture: tau_0 = M (Kp (q_ref - q) - Kd qd), projected with N^T = I - J^T Jbar^T
  const VecN tau_posture = d.M * (g.kp_posture_per_s2 * (r.q - c.q) - g.kd_posture_per_s * c.qd);
  tau_posture_null_space = tau_posture - d.J.transpose() * (Jbar_transpose * tau_posture);
}

namespace {
Vec3 orientationCommand(const ControlContext& c, const Reference& r, const TaskGains& g, const Vec6& twist) {
  return r.tool_angular_acceleration + g.kp_orientation_per_s2 * rotationError(r.tool_rotation, c.dyn.tool_rotation) +
         g.kd_orientation_per_s * (r.tool_angular_velocity - twist.tail<3>());
}
Vec3 positionCommand(const ControlContext& c, const Reference& r, const TaskGains& g, const Vec6& twist) {
  return r.tool_acceleration + g.kp_position_per_s2 * (r.tool_position - c.dyn.tool_position) +
         g.kd_position_per_s * (r.tool_velocity - twist.head<3>());
}
}  // namespace

VecN OperationalSpace::compute(const ControlContext& c, const Reference& r) {
  osc_.compute(c, r, g_);
  Vec6 commanded_acceleration;
  commanded_acceleration << positionCommand(c, r, g_, osc_.tool_twist), orientationCommand(c, r, g_, osc_.tool_twist);
  const Vec6 wrench = osc_.Lambda * (commanded_acceleration - c.dyn.Jdqd);
  return VecN(c.dyn.J.transpose() * wrench + c.dyn.h + osc_.tau_posture_null_space);
}

VecN CartesianImpedance::compute(const ControlContext& c, const Reference& r) {
  osc_.compute(c, r, g_);
  const Vec3 position_error = r.tool_position - c.dyn.tool_position;
  const Vec3 orientation_error = rotationError(r.tool_rotation, c.dyn.tool_rotation);
  Vec6 wrench;
  // Floors on the apparent task inertia (kg, kg m^2). Near a singularity a diagonal entry of
  // Lambda can collapse toward zero, which would remove the damping on that axis entirely.
  constexpr double kMinLinearInertiaKg = 1e-3, kMinRotationalInertiaKgM2 = 1e-5;
  for (int i = 0; i < 3; ++i) {
    // damping from the apparent task inertia so each axis gets roughly the requested ratio
    const double d_lin = 2.0 * p_.damping_ratio * std::sqrt(p_.stiffness_n_per_m[i] * std::max(osc_.Lambda(i, i), kMinLinearInertiaKg));
    const double d_rot =
        2.0 * p_.damping_ratio * std::sqrt(p_.stiffness_nm_per_rad[i] * std::max(osc_.Lambda(3 + i, 3 + i), kMinRotationalInertiaKgM2));
    wrench[i] = p_.stiffness_n_per_m[i] * position_error[i] + d_lin * (r.tool_velocity[i] - osc_.tool_twist[i]);
    wrench[3 + i] = p_.stiffness_nm_per_rad[i] * orientation_error[i] +
                    d_rot * (r.tool_angular_velocity[i] - osc_.tool_twist[3 + i]);
  }
  return VecN(c.dyn.J.transpose() * wrench + c.dyn.h + osc_.tau_posture_null_space);
}

void HybridForceMotion::reset() {
  force_integral_n_ = 0.0;
  push_force_filtered_n_ = 0.0;
  friction_coefficient_estimate_ = 0.0;
  in_contact_ = false;
}

VecN HybridForceMotion::compute(const ControlContext& c, const Reference& r) {
  osc_.compute(c, r, g_);
  const Vec3 normal = r.push_direction.normalized();
  const Mat3 motion_projector = Mat3::Identity() - normal * normal.transpose();
  const double push_measured = -c.tool_contact_force.dot(normal);  // the surface pushes back along -n
  push_force_filtered_n_ += lowPassGain(p_.force_filter_cutoff_hz, c.dt_s) * (push_measured - push_force_filtered_n_);
  if (push_force_filtered_n_ > p_.contact_detect_force_n) in_contact_ = true;  // latched: first touch ends the approach
  if (!in_contact_) return approach(c, r, motion_projector, normal);

  Vec6 commanded_acceleration;
  commanded_acceleration << motion_projector * positionCommand(c, r, g_, osc_.tool_twist),
      orientationCommand(c, r, g_, osc_.tool_twist);
  Vec6 wrench = osc_.Lambda * (commanded_acceleration - c.dyn.Jdqd);
  double normal_force = r.push_force_n;
  if (p_.use_force_feedback) {
    const double error = r.push_force_n - push_force_filtered_n_;
    force_integral_n_ = std::clamp(force_integral_n_ + p_.force_ki_per_s * error * c.dt_s, -p_.integrator_limit_n,
                                   p_.integrator_limit_n);
    normal_force += p_.force_kp * error + force_integral_n_;
  }
  normal_force -= p_.normal_damping_ns_per_m * normal.dot(osc_.tool_twist.head<3>());
  wrench.head<3>() = motion_projector * wrench.head<3>() + normal * normal_force;
  if (p_.compensate_surface_friction) compensateSurfaceFriction(c, r, motion_projector, wrench);
  return VecN(c.dyn.J.transpose() * wrench + c.dyn.h + osc_.tau_posture_null_space);
}

VecN HybridForceMotion::approach(const ControlContext& c, const Reference& r, const Mat3& motion_projector,
                                 const Vec3& normal) {
  // Velocity servo along the normal through the full OSC, so Lambda scales it. The first version
  // commanded a bare force of kv * v_approach, about 1 N, which could not break the 0.3 N m joint
  // stiction and the tool never reached the surface.
  Vec6 commanded_acceleration;
  commanded_acceleration << motion_projector * positionCommand(c, r, g_, osc_.tool_twist) +
                                normal * (g_.kd_position_per_s * (p_.approach_speed_m_per_s - normal.dot(osc_.tool_twist.head<3>()))),
      orientationCommand(c, r, g_, osc_.tool_twist);
  const Vec6 wrench = osc_.Lambda * (commanded_acceleration - c.dyn.Jdqd);
  return VecN(c.dyn.J.transpose() * wrench + c.dyn.h + osc_.tau_posture_null_space);
}

void HybridForceMotion::compensateSurfaceFriction(const ControlContext& c, const Reference& r, const Mat3& motion_projector,
                                                  Vec6& wrench) {
  // Coulomb drag on the surface opposes the motion and the position loop only reacts to it after
  // an error builds up. Estimate mu from the F/T while sliding (tangential over normal force) and
  // push ahead along the reference velocity by mu * f_n.
  const Vec3 tangential_force = motion_projector * c.tool_contact_force;
  const Vec3 tangential_velocity = motion_projector * osc_.tool_twist.head<3>();
  if (push_force_filtered_n_ > p_.contact_detect_force_n && tangential_velocity.norm() > p_.min_sliding_speed_m_per_s) {
    const double mu_sample = std::min(tangential_force.norm() / push_force_filtered_n_, 1.0);
    friction_coefficient_estimate_ += lowPassGain(p_.friction_estimate_cutoff_hz, c.dt_s) * (mu_sample - friction_coefficient_estimate_);
  }
  // Below 1 mm/s the reference direction is noise, and pushing along it would jitter.
  constexpr double kMinReferenceSpeedMPerS = 1e-3;
  const Vec3 reference_velocity = motion_projector * r.tool_velocity;
  if (reference_velocity.norm() > kMinReferenceSpeedMPerS)
    wrench.head<3>() += reference_velocity.normalized() * (friction_coefficient_estimate_ * std::max(push_force_filtered_n_, 0.0));
}

}  // namespace g1rtc
