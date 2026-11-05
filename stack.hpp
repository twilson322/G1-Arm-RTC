// The 1 kHz control stack: everything that runs in the control thread, in one object.
//
//   sensing    F/T compensation, dynamics cache, momentum observer
//   monitoring contact-aware collision detection and isolation
//   tasks      pick up planner output from the mailbox, check it starts where the robot is
//   supervisor decide the mode
//   control    run the controller for that mode, saturate
//   adaptation optional payload estimation written back into the model
//   logging    fixed-size record into a lock-free queue
//
// The same object runs in the deterministic lockstep experiments, the threaded runtime and the
// ROS 2 node. tick() does not allocate, lock or perform I/O.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "g1rtc/controllers.hpp"
#include "g1rtc/observer.hpp"
#include "g1rtc/robot_description.hpp"
#include "g1rtc/robot_sim.hpp"
#include "g1rtc/rt.hpp"
#include "g1rtc/supervisor.hpp"
#include "g1rtc/tasks.hpp"

namespace g1rtc {

/// Hands tasks from non-real-time threads to the control thread. The control side never locks or
/// frees: a task is released only once the control thread has reported, through two atomics, that
/// it can no longer be using it (a small RCU-style deferred reclamation).
class TaskMailbox {
 public:
  /// Any non-RT thread. Returns the sequence number of the posted task.
  std::uint64_t post(std::shared_ptr<const Task> task);
  /// Control thread: the newest posted task (or nullptr) and its sequence number.
  const Task* newest(std::uint64_t& seq);
  /// Control thread: the sequence number it is executing, 0 for none.
  void reportExecuting(std::uint64_t seq) { executing_seq_.store(seq, std::memory_order_release); }
  /// Any non-RT thread: free tasks the control thread can no longer reference. Returns how many.
  std::size_t collect();
  std::size_t retained() const;

 private:
  struct Slot {
    const Task* task = nullptr;
    std::uint64_t seq = 0;
  };
  mutable std::mutex owners_mutex_;  // only ever taken by non-RT threads
  std::deque<std::pair<std::uint64_t, std::shared_ptr<const Task>>> owners_;
  std::uint64_t next_seq_ = 0;
  rt::TripleBuffer<Slot> published_;
  Slot newest_seen_{};
  std::atomic<std::uint64_t> newest_seen_seq_{0}, executing_seq_{0};
};

struct StackConfig {
  SupervisorConfig supervisor;
  double nominal_period_s = 0.001;  // used until two state samples give a measured period

  MomentumObserver::Params observer;
  bool detect_collisions = true;
  CollisionDetector::Params detector;
  /// The detector re-arms after the residual has stayed below this fraction of the thresholds
  /// for `rearm_quiet_s` in Hold, so one contact cannot trigger a second reflex.
  double rearm_fraction = 0.5;
  double rearm_quiet_s = 0.2;
  std::vector<std::string> contact_candidates_proximal_to_distal = {
      "torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
      "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link", "tool"};

  double tool_mass_kg = 0.15;  // what the F/T compensation subtracts

  bool estimate_payload = false;
  bool adapt_model_to_payload = false;
  PayloadEstimator::Params payload;
  double payload_model_update_period_s = 0.2;
  double max_payload_kg = 3.0;
  /// After an observer reset the residual needs about five time constants (5 / 30 s) to settle;
  /// payload samples taken earlier are skipped.
  double observer_settle_s = 0.15;

  // a task may only start where the robot already is
  double start_tolerance_joint_rad = 0.05;
  double start_tolerance_position_m = 0.02;
  double start_tolerance_orientation_rad = 0.05;

  double reflex_residual_gain = 1.0;  // Retract: tau = g + K r - D qd
  VecN reflex_damping_nms_per_rad;    // also the SafeStop damping

  JointPdGravity hold = JointPdGravity::defaultsG1();
  double computed_torque_bandwidth_hz = 6.0, computed_torque_damping_ratio = 0.9;
  TaskGains task_gains;
  ImpedanceParams impedance;
  HybridParams hybrid;

  static StackConfig defaultsG1();
};

/// Fixed-size record for the lock-free logger queue (floats keep it under 0.5 kB).
struct LogRecord {
  double time_s = 0.0;
  std::uint8_t mode = 0, saturated_joints = 0, tracking_mode = 0;
  float q[kMaxDof]{}, qd[kMaxDof]{}, q_ref[kMaxDof]{}, tau[kMaxDof]{}, residual[kMaxDof]{};
  float tool_position[3]{}, tool_position_ref[3]{}, tool_contact_force[3]{};
  float tool_orientation_wxyz[4]{1.f, 0.f, 0.f, 0.f};
  float push_force_filtered_n = 0.f, push_force_ref_n = 0.f;
  float payload_estimate_kg = 0.f;
};
using LogQueue = rt::SpscQueue<LogRecord, 8192>;

class ControlStack {
 public:
  ControlStack(const RobotDescription& robot, StackConfig cfg);
  ControlStack(const ControlStack&) = delete;
  ControlStack& operator=(const ControlStack&) = delete;

  /// One control period on the latest robot state.
  TorqueCommand tick(const RobotState& state);

  Supervisor& supervisor() { return supervisor_; }
  const Supervisor& supervisor() const { return supervisor_; }
  TaskMailbox& mailbox() { return mailbox_; }
  void attachLog(LogQueue* queue) { log_queue_ = queue; }
  const LogRecord& lastRecord() const { return record_; }

  const VecN& residual() const { return observer_.residual(); }
  /// Residual minus the (lagged) joint torque explained by the measured tool force.
  const VecN& unexplainedResidual() const { return unexplained_residual_; }
  const Vec3& toolContactForce() const { return tool_contact_force_; }
  const rbd::DynamicsCache& dynamics() const { return dyn_; }
  const rbd::Model& model() const { return model_; }
  const PayloadEstimator& payloadEstimator() const { return payload_; }
  bool collisionDetectedThisTick() const { return collision_this_tick_; }
  /// Body the last detected collision was attributed to (model body id), -1 if none.
  int isolatedBody() const { return isolated_body_; }
  int saturatedJoints() const { return saturated_joints_; }
  double taskTime() const { return task_time_s_; }
  double lastPeriod() const { return period_s_; }

 private:
  void updateSensing(const RobotState& state, Mode mode_before);
  bool monitorCollisions(Mode mode_before);
  bool pendingTaskStartsHere(Mode mode_before);
  void switchTask(Mode before, Mode after);
  Controller& controllerFor(TrackingMode mode);
  VecN computeTorque(Mode mode);
  void updatePayloadEstimate(Mode mode);
  void writeLogRecord(Mode mode, const VecN& tau);

  RobotDescription robot_;
  rbd::Model& model_;  // robot_.model, adapted when the payload estimate is applied
  rbd::Model nominal_model_;
  StackConfig cfg_;
  rbd::DynamicsCache dyn_;
  MomentumObserver observer_;
  CollisionDetector detector_;
  ToolWrenchEstimator wrench_;
  PayloadEstimator payload_;
  Eigen::Vector4d payload_in_model_ = Eigen::Vector4d::Zero();
  Supervisor supervisor_;
  TaskMailbox mailbox_;
  JointPdGravity hold_;
  ComputedTorque computed_torque_;
  OperationalSpace operational_space_;
  CartesianImpedance impedance_;
  HybridForceMotion hybrid_;
  std::vector<int> contact_candidates_;

  // per-tick state
  double time_s_ = 0.0, period_s_ = 0.0;
  bool have_previous_state_ = false;
  VecN q_, qd_, previous_impulse_, mean_applied_torque_;
  Vec3 tool_contact_force_{0, 0, 0};
  Mat3 ft_sensor_rotation_ = Mat3::Identity();
  bool observer_running_ = false;
  double observer_started_s_ = 0.0;
  VecN unexplained_residual_, contact_torque_lagged_;
  bool collision_this_tick_ = false;
  double quiet_since_s_ = 0.0;
  int isolated_body_ = -1;
  const Task* pending_task_ = nullptr;
  const Task* active_task_ = nullptr;
  std::uint64_t pending_seq_ = 0, active_seq_ = 0;
  double task_start_s_ = 0.0, task_time_s_ = 0.0, last_model_update_s_ = 0.0;
  int saturated_joints_ = 0;
  Reference reference_;
  LogRecord record_;
  LogQueue* log_queue_ = nullptr;
};

}  // namespace g1rtc
