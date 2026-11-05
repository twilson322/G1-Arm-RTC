// The "robot": MuJoCo at 2 kHz behind an interface that exposes only what real hardware would.
//
// The controller never sees MuJoCo's state. It gets:
//   * joint positions quantized to the encoder resolution,
//   * joint velocities differenced from those positions at 2 kHz and low-pass filtered (how many
//     joint drives estimate speed),
//   * the wrist force/torque reading, with noise, including the tool's own weight,
//   * the running integral of the torque each drive applied, which is what lets the momentum
//     observer integrate correctly when the control thread's period jitters (see RobotState).
// Torque commands pass through a transport delay, the drive's current limit and an optional
// torque-constant error, and the joints have Coulomb friction the controller only partly models.
// Each drive also runs a watchdog: if commands stop arriving it switches to damping and latches
// a fault until a command carries the reset flag.
//
// Ground truth (true state, contact and disturbance torques) is exposed separately and is only
// used by experiments and tests to score the estimators.
//
// Not thread-safe: in the threaded runtime only the robot thread touches a RobotSim.
#pragma once

#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include "g1rtc/rbd.hpp"

struct mjData_;

namespace g1rtc {

struct SimConfig {
  // 2^18 counts per output revolution: a 14-bit motor encoder read through a ~16:1 reduction,
  // which is how G1-class actuators sense joint position.
  double encoder_counts_per_rev = 262144.0;
  double velocity_filter_cutoff_hz = 150.0;  // first order; 0 passes the differenced velocity through
  double ft_force_noise_std_n = 0.3;
  double ft_torque_noise_std_nm = 0.01;
  int command_delay_physics_steps = 1;    // 0.5 ms each
  double torque_constant_error = 0.0;     // each joint's Kt is off by U(-e, e), as a fraction
  double watchdog_timeout_s = 0.005;
  double fault_damping_nms_per_rad = 3.0;
  bool joint_friction = true;             // the G1 model's 0.3 N m Coulomb friction per joint
  // A stiff sphere on a rigid table chattered in and out of contact while sliding; a softer
  // contact (longer time constant) behaves like the compliant sponge the tool is meant to be.
  double sponge_contact_timeconst_s = 0.05;
  bool elliptic_friction_cones = true;    // pyramidal cones also chattered when sliding
};

/// Everything the drives and sensors report in one 2 kHz cycle.
struct RobotState {
  double time_s = 0.0;
  VecN q, qd;  // rad, rad/s (quantized and filtered, see SimConfig)
  /// Wrist F/T reading [force N; torque N m] in the sensor frame, which coincides with the tool
  /// frame. It includes the tool's weight and an unknown offset; see ToolWrenchEstimator.
  Vec6 wrist_ft_reading = Vec6::Zero();
  /// Integral of the torque each drive applied since reset, N m s. The drive knows the current it
  /// drove but not its true torque constant, so this is the commanded torque after the delay,
  /// the current limit and the fault logic, and it excludes SimConfig::torque_constant_error.
  VecN applied_torque_impulse;
  bool drive_fault = false;
  std::uint64_t last_command_seq = 0;
};

struct TorqueCommand {
  VecN tau;  // N m
  std::uint64_t seq = 0;
  bool reset_fault = false;
};

/// A scheduled force on one body, used to inject collisions with known ground truth.
struct Disturbance {
  int body = -1;
  Vec3 force_world_n{0, 0, 0};  // applied at the body COM
  double start_s = 0.0;
  double duration_s = 0.0;
};

class RobotSim {
 public:
  RobotSim(const std::string& xml_path, const SimConfig& cfg, std::uint64_t seed);
  ~RobotSim();
  RobotSim(const RobotSim&) = delete;
  RobotSim& operator=(const RobotSim&) = delete;

  void reset(const VecN& q0);
  /// Hand a command to the drives. It reaches the motors `command_delay_physics_steps` later.
  void command(const TorqueCommand& cmd);
  /// One physics step including the drive, watchdog and sensor models.
  void step();
  const RobotState& state() const { return state_; }
  double physicsDt() const;
  double time() const;
  int nv() const { return nv_; }
  const mjModel_* mjModel() const { return m_; }
  mjData_* mjData() { return d_; }

  // experiment hooks
  void scheduleDisturbance(const Disturbance& d) { disturbances_.push_back(d); }
  /// Rigidly attach an unmodelled point mass to the tool (COM in the tool frame, m).
  void setPayload(double mass_kg, const Vec3& com_tool_m);
  void setIntruderPose(const Vec3& position_world_m, const Mat3& rotation_world);
  /// Move the table top vertically (surface-height uncertainty experiments).
  void setTableHeightOffset(double offset_m);
  int bodyId(const char* name) const;

  // ground truth, not available to the controller
  VecN trueQ() const;
  VecN trueQd() const;
  /// Joint torques from contacts plus scheduled disturbances: what a collision detector should see.
  const VecN& trueExternalTorque() const { return tau_external_true_; }
  /// Net contact force of the environment on the tool sponge, world frame, N.
  const Vec3& toolContactForce() const { return tool_contact_force_; }
  /// Sum of contact force magnitudes between the robot and the named environment geom, N.
  double contactForceWith(const char* env_geom) const;

 private:
  int requireId(int type, const char* name) const;
  void applyDriveModel(double t);
  void applyDisturbances(double t);
  void updateGroundTruth();
  void updateSensors();

  mjModel_* m_ = nullptr;
  mjData_* d_ = nullptr;
  SimConfig cfg_;
  int nv_ = 0;
  std::mt19937_64 rng_;
  std::normal_distribution<double> unit_normal_{0.0, 1.0};
  RobotState state_;
  std::vector<int> actuator_of_dof_;
  VecN q_quantized_prev_, qd_filtered_, torque_constant_scale_, tau_drive_, tau_external_true_;
  std::deque<std::pair<double, TorqueCommand>> commands_in_flight_;  // (time it reaches the motor, command)
  TorqueCommand active_command_;
  double last_command_time_ = -1.0;
  bool fault_ = false;
  bool have_command_ = false;
  std::vector<Disturbance> disturbances_;
  int tool_body_ = -1, sponge_geom_ = -1, intruder_body_ = -1, table_body_ = -1;
  int ft_force_adr_ = -1, ft_torque_adr_ = -1;
  Vec3 tool_contact_force_{0, 0, 0};
  double base_tool_mass_ = 0.0, table_height0_ = 0.0;
  Vec3 base_tool_com_{0, 0, 0};
  // scratch for updateGroundTruth (the plant may allocate, but there is no reason to every step)
  std::vector<double> efc_contact_force_, joint_scratch_a_, joint_scratch_b_;
};

}  // namespace g1rtc
