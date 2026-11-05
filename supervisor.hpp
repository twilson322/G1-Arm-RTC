// The safety layer: a small state machine that owns the operating mode and can always fall back
// to a state that does not depend on the planner, the task or the estimated parts.
//
//   Disabled --enable--> Enabling --(drive enabled, F/T tared)--> Hold
//   Hold --start(task)--> Executing --(task done)--> Hold
//   Hold/Executing --collision--> Reflex --(reflex duration elapsed)--> Hold at the current pose
//   any active mode --drive fault | joint-limit or speed violation | stop--> SafeStop --reset--> Enabling
//
// Hold is bumpless: its setpoint is the measured pose when it is entered, so switching into it
// never produces a torque step. Transitions go to a fixed-size ring so the control thread never
// allocates. Operator requests and the current mode are atomics so other threads can use them;
// everything else belongs to the control thread.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

#include "g1rtc/rbd.hpp"

namespace g1rtc {

enum class Mode : std::uint8_t { Disabled, Enabling, Hold, Executing, Reflex, SafeStop };
enum class ReflexKind : std::uint8_t { Ignore, Stop, Float, Retract };
enum class Reason : std::uint8_t {
  None, EnableRequest, DriveEnabled, StartRequest, TaskDone, Collision, ReflexDone, DriveFault,
  JointLimit, SpeedLimit, StopRequest, ResetRequest, StartRejected
};
const char* toString(Mode m);
const char* toString(Reason r);
const char* toString(ReflexKind k);

struct SupervisorConfig {
  ReflexKind reflex = ReflexKind::Retract;
  double reflex_duration_s = 0.4;
  double joint_limit_margin_rad = 0.01;
  /// Inside the margin only motion toward the limit trips. Holding still there must be allowed:
  /// after a drive fault the arm can sag onto a joint stop, and an absolute check then tripped
  /// again on the first Hold tick, so the robot could never be re-enabled.
  double toward_limit_speed_rad_s = 0.02;
  double speed_limit_scale = 1.0;  // trip at scale * model qd_max
  double enable_duration_s = 0.05;  // drive reset round trip plus F/T taring
};

struct Transition {
  double time_s = 0.0;
  Mode from = Mode::Disabled, to = Mode::Disabled;
  Reason why = Reason::None;
};

class Supervisor {
 public:
  Supervisor(const rbd::Model& m, SupervisorConfig cfg) : model_(&m), cfg_(cfg), hold_pose_(VecN::Zero(m.nv)) {}
  Supervisor(const Supervisor&) = delete;
  Supervisor& operator=(const Supervisor&) = delete;

  void requestEnable() { enable_requested_.store(true, std::memory_order_release); }
  void requestStart() { start_requested_.store(true, std::memory_order_release); }
  void requestStop() { stop_requested_.store(true, std::memory_order_release); }
  void requestReset() { reset_requested_.store(true, std::memory_order_release); }

  struct Inputs {
    double time_s;
    const VecN& q;
    const VecN& qd;
    bool drive_fault;
    bool collision_detected;  // the detector fired this tick
    bool task_finished;
    bool task_ready_to_start;  // a task is pending and begins where the robot is
  };
  /// Advance one tick and return the mode to act in.
  Mode step(const Inputs& in);

  Mode mode() const { return mode_.load(std::memory_order_acquire); }
  /// Why the current mode was entered. Safe from any thread, unlike transition().
  Reason lastReason() const { return last_reason_.load(std::memory_order_acquire); }
  double modeEntryTime() const { return mode_entry_time_s_; }
  const VecN& holdPose() const { return hold_pose_; }
  bool wantsDriveReset() const { return mode() == Mode::Enabling; }
  const SupervisorConfig& config() const { return cfg_; }
  std::size_t transitionCount() const { return transition_count_; }
  /// i-th transition still in the ring, oldest first (the ring keeps the last 64). Control thread
  /// only, or after it has stopped: the ring is not synchronized.
  const Transition& transition(std::size_t i) const;
  std::size_t retainedTransitions() const { return std::min(transition_count_, transitions_.size()); }

 private:
  void enter(Mode to, Reason why, double time_s, const VecN& q);
  void record(Mode from, Mode to, Reason why, double time_s);
  Reason limitViolation(const VecN& q, const VecN& qd) const;

  const rbd::Model* model_;
  SupervisorConfig cfg_;
  std::atomic<Mode> mode_{Mode::Disabled};
  std::atomic<Reason> last_reason_{Reason::None};
  double mode_entry_time_s_ = 0.0;
  VecN hold_pose_;
  std::atomic<bool> enable_requested_{false}, start_requested_{false}, stop_requested_{false}, reset_requested_{false};
  std::array<Transition, 64> transitions_{};
  std::size_t transition_count_ = 0;
};

}  // namespace g1rtc
