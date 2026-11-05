// Proprioceptive estimation: no joint torque sensors, only encoders, the torque the drives report
// they applied, the wrist F/T sensor and the model.
//
// MomentumObserver (De Luca et al.): with generalized momentum p = M(q) qd,
//     p_dot = tau + C^T(q, qd) qd - g(q) + tau_ext,
// the residual r = K (p - p0 - integral(tau - beta + r) dt), beta = g - C^T qd, obeys
// r_dot = K (tau_ext - r): a first-order low-pass estimate of the external joint torque that needs
// neither joint accelerations nor M^-1. Anything the model gets wrong (unmodelled friction,
// payload, torque-constant error, sensing delay) also lands in r, and the collision thresholds
// have to sit above it.
#pragma once

#include <vector>

#include "g1rtc/rbd.hpp"

namespace g1rtc {

class MomentumObserver {
 public:
  struct Params {
    VecN gain_per_s;
    bool model_joint_friction = true;
    double coulomb_friction_nm = 0.3;   // what the friction model subtracts per joint
    double friction_smoothing_rad_s = 0.02;  // tanh width standing in for sign(qd)
    /// The drives low-pass their velocity estimate, so p = M qd_filtered lags the true momentum.
    /// Passing tau and beta through the same first-order filter puts both sides of
    /// p_dot = tau - beta + tau_ext on the same delay. 0 disables (ideal velocity sensing).
    double drive_velocity_filter_hz = 150.0;
  };
  MomentumObserver(const rbd::Model& m, Params p);
  void reset(const VecN& q, const VecN& qd);
  /// `mean_applied_torque` is the average torque the drives applied since the previous update,
  /// and `dt_s` the time between the two state samples.
  const VecN& update(const VecN& q, const VecN& qd, const VecN& mean_applied_torque, double dt_s);
  const VecN& residual() const { return residual_; }

 private:
  const rbd::Model* model_;
  Params p_;
  VecN momentum_integral_, residual_, tau_filtered_, beta_filtered_, beta_previous_;
  bool filters_initialized_ = false;
  rbd::Kinematics<double> kin_;
  rbd::Workspace<double> work_;
  rbd::Kinematics<Dual> kin_dual_;
  rbd::Workspace<Dual> work_dual_;
  MatN M_;
};

/// Per-joint thresholds on |r| with a persistence requirement. Fires once, then stays tripped
/// until rearm().
class CollisionDetector {
 public:
  struct Params {
    VecN threshold_nm;
    int persistence_ticks = 3;
  };
  explicit CollisionDetector(Params p) : p_(std::move(p)) {}
  /// Returns true on the sample where a collision is declared.
  bool update(const VecN& residual);
  void rearm();
  bool tripped() const { return tripped_; }
  const Params& params() const { return p_; }

 private:
  Params p_;
  int consecutive_ = 0;
  bool tripped_ = false;
};

/// Contact-link isolation by residual fitting. For each candidate body, fit a point force at its
/// COM so that r ~ J^T f (least squares) and score the relative fit error. A contact on link k
/// leaves the joints distal to k untouched, so a too-proximal candidate cannot explain r; a
/// too-distal one usually explains it worse. Of the candidates whose error is within
/// `tie_tolerance` of the best, the most proximal wins (candidates must be ordered proximal to
/// distal). Each joint is divided by `joint_scale` (its detection threshold) so small wrist
/// torques count as much as large waist ones. A contact point at the COM is exact for the
/// injected test disturbances and an approximation for real contacts on a link's surface.
inline constexpr std::size_t kMaxContactCandidates = 16;
int isolateContactBody(const rbd::Model& m, const rbd::Kinematics<double>& k, const VecN& residual, const VecN& joint_scale,
                       const std::vector<int>& candidates_proximal_to_distal, double tie_tolerance = 0.05);

/// Turns the raw wrist F/T reading into the force the environment exerts on the tool (world
/// frame), using the sensor orientation from the controller's own kinematics and removing the
/// tool's weight. Tool inertial forces are ignored: 0.15 kg times the accelerations here is under
/// 1 N.
class ToolWrenchEstimator {
 public:
  ToolWrenchEstimator(double tool_mass_kg, const Vec3& gravity) : tool_mass_kg_(tool_mass_kg), gravity_(gravity) {}
  Vec3 contactForceWorld(const Vec6& reading, const Mat3& sensor_rotation_world) const;
  /// Average readings taken with the tool in free space and at rest into the sensor offset.
  void accumulateTare(const Vec6& reading, const Mat3& sensor_rotation_world);
  void clearTare();

 private:
  double tool_mass_kg_;
  Vec3 gravity_;
  Vec3 force_offset_sensor_{0, 0, 0};
  Vec3 offset_sum_{0, 0, 0};
  int offset_samples_ = 0;
};

/// Recursive least squares on the observer residual for an unknown payload held in the hand.
/// Quasi-static model: the payload's weight enters the joints as r = Y(q) theta with
///   theta = [m, m c_x, m c_y, m c_z]   (c: payload COM in the tool frame)
///   Y     = [J_v^T g,  -J_w^T [g]x R_tool]
/// Payload inertia and residual friction are not in the regressor; both bias the estimate. The COM
/// is only identifiable when the tool orientation varies, so `estimate_com = false` fits the mass
/// alone with the COM at the tool frame.
class PayloadEstimator {
 public:
  struct Params {
    double forgetting = 0.99998;  // applied per joint update, nv per tick: about 5 s memory at 1 kHz
    double initial_covariance = 100.0;
    bool estimate_com = true;
    double max_com_offset_m = 0.15;  // a hand-held payload's COM is near the hand
  };
  explicit PayloadEstimator(Params p);
  void reset();
  using Regressor = Eigen::Matrix<double, Eigen::Dynamic, 4, 0, kMaxDof, 4>;
  static Regressor regressor(const rbd::DynamicsCache& dyn, const Vec3& gravity);
  /// `already_modelled` is the payload the controller model already contains (and that is
  /// therefore missing from the residual); it is added back so the estimate is of the whole payload.
  void update(const rbd::DynamicsCache& dyn, const VecN& residual, const Vec3& gravity,
              const Eigen::Vector4d& already_modelled = Eigen::Vector4d::Zero());
  double massKg() const { return theta_[0]; }
  Vec3 comToolFrame() const;
  const Eigen::Vector4d& theta() const { return theta_; }

 private:
  Params p_;
  Eigen::Vector4d theta_;
  Eigen::Matrix4d P_;
};

}  // namespace g1rtc
