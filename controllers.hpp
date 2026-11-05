// Torque controllers for the 10-DOF upper body. All read the DynamicsCache computed once per tick
// from the controller's own model (not the simulator's) and return joint torques.
//
//  JointPdGravity      tau = Kp e + Kd e_dot + g(q)                         baseline
//  ComputedTorque      tau = M (qdd_ref + Kp e + Kd e_dot) + h(q, qd)       joint-space inverse dynamics
//  OperationalSpace    tau = J^T Lambda (a* - J_dot qd) + h + N^T tau_0     Khatib OSC on the 6-D tool
//                                                                         pose; 4-D posture null space
//  CartesianImpedance  tau = J^T (K e - D v) + h + N^T tau_0                no inertia shaping
//  HybridForceMotion   OSC in the plane of the surface, force PI along its normal
//
// Nothing here allocates: every DOF-sized object is a fixed-capacity Eigen type.
#pragma once

#include <cmath>
#include <string_view>

#include "g1rtc/rbd.hpp"

namespace g1rtc {

struct Reference {
  VecN q, qd, qdd;  // joint reference; also the null-space posture of the task-space controllers
  Vec3 tool_position{0, 0, 0}, tool_velocity{0, 0, 0}, tool_acceleration{0, 0, 0};
  Mat3 tool_rotation = Mat3::Identity();
  Vec3 tool_angular_velocity{0, 0, 0}, tool_angular_acceleration{0, 0, 0};
  Vec3 push_direction{0, 0, -1};  // HybridForceMotion: the tool pushes the surface along this
  double push_force_n = 0.0;
  static Reference hold(const VecN& q);
};

struct ControlContext {
  const rbd::Model& model;
  const rbd::DynamicsCache& dyn;
  const VecN& q;
  const VecN& qd;
  const Vec3& tool_contact_force;  // environment on tool, world frame, tool weight removed
  double dt_s;
};

class Controller {
 public:
  virtual ~Controller() = default;
  virtual VecN compute(const ControlContext& c, const Reference& r) = 0;
  virtual void reset() {}
  virtual std::string_view name() const = 0;
};

class JointPdGravity final : public Controller {
 public:
  JointPdGravity(VecN kp_nm_per_rad, VecN kd_nms_per_rad) : kp_(std::move(kp_nm_per_rad)), kd_(std::move(kd_nms_per_rad)) {}
  VecN compute(const ControlContext& c, const Reference& r) override;
  std::string_view name() const override { return "joint_pd_gravity"; }
  static JointPdGravity defaultsG1();

 private:
  VecN kp_, kd_;
};

/// Same natural frequency and damping ratio on every joint, which inverse dynamics makes possible.
class ComputedTorque final : public Controller {
 public:
  ComputedTorque(double bandwidth_hz, double damping_ratio);
  VecN compute(const ControlContext& c, const Reference& r) override;
  std::string_view name() const override { return "computed_torque"; }

 private:
  double kp_per_s2_, kd_per_s_;
};

struct TaskGains {
  // tool-space stiffness and damping as accelerations: about 4.8 Hz with damping ratio 0.9
  double kp_position_per_s2 = 900.0, kd_position_per_s = 55.0;
  double kp_orientation_per_s2 = 600.0, kd_orientation_per_s = 45.0;
  double kp_posture_per_s2 = 80.0, kd_posture_per_s = 18.0;
  // Tikhonov term on Lambda^-1 so the task inertia stays bounded near singular postures
  double lambda_regularization = 1e-3;
};

/// Shared operational-space algebra: Lambda, the dynamically consistent null-space projector and
/// the posture torque.
struct OscTerms {
  Mat6 Lambda;
  Mat6N Jbar_transpose;  // (M^-1 J^T Lambda)^T
  VecN tau_posture_null_space;
  Vec6 tool_twist;  // J qd: [linear; angular]
  void compute(const ControlContext& c, const Reference& r, const TaskGains& g);
};

class OperationalSpace final : public Controller {
 public:
  explicit OperationalSpace(TaskGains g = {}) : g_(g) {}
  VecN compute(const ControlContext& c, const Reference& r) override;
  std::string_view name() const override { return "operational_space"; }

 private:
  TaskGains g_;
  OscTerms osc_;
};

struct ImpedanceParams {
  Vec3 stiffness_n_per_m{600, 600, 600};
  Vec3 stiffness_nm_per_rad{30, 30, 30};
  double damping_ratio = 0.8;
};

class CartesianImpedance final : public Controller {
 public:
  explicit CartesianImpedance(ImpedanceParams p = {}, TaskGains g = {}) : p_(p), g_(g) {}
  VecN compute(const ControlContext& c, const Reference& r) override;
  std::string_view name() const override { return "cartesian_impedance"; }

 private:
  ImpedanceParams p_;
  TaskGains g_;
  OscTerms osc_;
};

struct HybridParams {
  double force_kp = 0.6;                 // N per N of force error
  double force_ki_per_s = 12.0;
  double integrator_limit_n = 15.0;
  double normal_damping_ns_per_m = 25.0;
  double force_filter_cutoff_hz = 40.0;  // the F/T noise is 0.3 N per sample
  bool use_force_feedback = true;        // false: feed-forward force only (open-loop J^T F)
  double approach_speed_m_per_s = 0.05;  // guarded approach until first contact
  double contact_detect_force_n = 1.5;
  bool compensate_surface_friction = false;
  double friction_estimate_cutoff_hz = 2.0;
  double min_sliding_speed_m_per_s = 0.01;
};

class HybridForceMotion final : public Controller {
 public:
  explicit HybridForceMotion(HybridParams p = {}, TaskGains g = {}) : p_(p), g_(g) {}
  VecN compute(const ControlContext& c, const Reference& r) override;
  void reset() override;
  std::string_view name() const override { return "hybrid_force_motion"; }
  double filteredPushForce() const { return push_force_filtered_n_; }
  bool inContact() const { return in_contact_; }
  double surfaceFrictionEstimate() const { return friction_coefficient_estimate_; }

 private:
  VecN approach(const ControlContext& c, const Reference& r, const Mat3& motion_projector, const Vec3& normal);
  void compensateSurfaceFriction(const ControlContext& c, const Reference& r, const Mat3& motion_projector, Vec6& wrench);

  HybridParams p_;
  TaskGains g_;
  OscTerms osc_;
  double force_integral_n_ = 0.0, push_force_filtered_n_ = 0.0, friction_coefficient_estimate_ = 0.0;
  bool in_contact_ = false;
};

/// Clamp to the model's torque limits; returns how many joints were clipped.
int saturate(const rbd::Model& m, VecN& tau);

/// Discrete first-order low-pass gain for cutoff f and step dt (exact for a constant input).
inline double lowPassGain(double cutoff_hz, double dt_s) {
  return 1.0 - std::exp(-2.0 * 3.141592653589793 * cutoff_hz * dt_s);
}

}  // namespace g1rtc
