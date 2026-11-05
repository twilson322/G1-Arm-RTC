#include "g1rtc/supervisor.hpp"

#include <cmath>

namespace g1rtc {

const char* toString(Mode m) {
  switch (m) {
    case Mode::Disabled: return "disabled";
    case Mode::Enabling: return "enabling";
    case Mode::Hold: return "hold";
    case Mode::Executing: return "executing";
    case Mode::Reflex: return "reflex";
    case Mode::SafeStop: return "safe_stop";
  }
  return "?";
}

const char* toString(Reason r) {
  switch (r) {
    case Reason::None: return "none";
    case Reason::EnableRequest: return "enable_request";
    case Reason::DriveEnabled: return "drive_enabled";
    case Reason::StartRequest: return "start_request";
    case Reason::TaskDone: return "task_done";
    case Reason::Collision: return "collision";
    case Reason::ReflexDone: return "reflex_done";
    case Reason::DriveFault: return "drive_fault";
    case Reason::JointLimit: return "joint_limit";
    case Reason::SpeedLimit: return "speed_limit";
    case Reason::StopRequest: return "stop_request";
    case Reason::ResetRequest: return "reset_request";
    case Reason::StartRejected: return "start_rejected";
  }
  return "?";
}

const char* toString(ReflexKind k) {
  switch (k) {
    case ReflexKind::Ignore: return "ignore";
    case ReflexKind::Stop: return "stop";
    case ReflexKind::Float: return "float";
    case ReflexKind::Retract: return "retract";
  }
  return "?";
}

const Transition& Supervisor::transition(std::size_t i) const {
  const std::size_t first = transition_count_ - retainedTransitions();
  return transitions_[(first + i) % transitions_.size()];
}

void Supervisor::record(Mode from, Mode to, Reason why, double time_s) {
  transitions_[transition_count_ % transitions_.size()] = Transition{time_s, from, to, why};
  ++transition_count_;
}

void Supervisor::enter(Mode to, Reason why, double time_s, const VecN& q) {
  record(mode(), to, why, time_s);
  last_reason_.store(why, std::memory_order_release);
  mode_.store(to, std::memory_order_release);
  mode_entry_time_s_ = time_s;
  hold_pose_ = q;
}

Reason Supervisor::limitViolation(const VecN& q, const VecN& qd) const {
  for (int i = 0; i < model_->nv; ++i) {
    const bool near_upper = q[i] > model_->q_max[i] - cfg_.joint_limit_margin_rad;
    const bool near_lower = q[i] < model_->q_min[i] + cfg_.joint_limit_margin_rad;
    if ((near_upper && qd[i] > cfg_.toward_limit_speed_rad_s) || (near_lower && qd[i] < -cfg_.toward_limit_speed_rad_s))
      return Reason::JointLimit;
    if (std::abs(qd[i]) > cfg_.speed_limit_scale * model_->qd_max[i]) return Reason::SpeedLimit;
  }
  return Reason::None;
}

Mode Supervisor::step(const Inputs& in) {
  const bool enable = enable_requested_.exchange(false, std::memory_order_acq_rel);
  const bool start = start_requested_.exchange(false, std::memory_order_acq_rel);
  const bool stop = stop_requested_.exchange(false, std::memory_order_acq_rel);
  const bool reset = reset_requested_.exchange(false, std::memory_order_acq_rel);
  const Mode current = mode();
  const bool torque_controlled = current == Mode::Hold || current == Mode::Executing || current == Mode::Reflex;

  // The monitors run before anything else and do not depend on the task or the planner.
  if (torque_controlled) {
    Reason fault = limitViolation(in.q, in.qd);
    if (fault == Reason::None && in.drive_fault) fault = Reason::DriveFault;
    if (fault == Reason::None && stop) fault = Reason::StopRequest;
    if (fault != Reason::None) {
      enter(Mode::SafeStop, fault, in.time_s, in.q);
      return mode();
    }
  }

  const bool react_to_collision = in.collision_detected && cfg_.reflex != ReflexKind::Ignore;
  switch (current) {
    case Mode::Disabled:
      if (enable) enter(Mode::Enabling, Reason::EnableRequest, in.time_s, in.q);
      break;
    case Mode::Enabling:
      if (!in.drive_fault && in.time_s - mode_entry_time_s_ >= cfg_.enable_duration_s)
        enter(Mode::Hold, Reason::DriveEnabled, in.time_s, in.q);
      break;
    case Mode::Hold:
      if (start) {
        if (in.task_ready_to_start)
          enter(Mode::Executing, Reason::StartRequest, in.time_s, in.q);
        else
          record(current, current, Reason::StartRejected, in.time_s);
      } else if (react_to_collision) {
        enter(Mode::Reflex, Reason::Collision, in.time_s, in.q);
      }
      break;
    case Mode::Executing:
      if (react_to_collision)
        enter(Mode::Reflex, Reason::Collision, in.time_s, in.q);
      else if (in.task_finished)
        enter(Mode::Hold, Reason::TaskDone, in.time_s, in.q);
      break;
    case Mode::Reflex:
      if (in.time_s - mode_entry_time_s_ >= cfg_.reflex_duration_s) enter(Mode::Hold, Reason::ReflexDone, in.time_s, in.q);
      break;
    case Mode::SafeStop:
      if (reset) enter(Mode::Enabling, Reason::ResetRequest, in.time_s, in.q);
      break;
  }
  return mode();
}

}  // namespace g1rtc
