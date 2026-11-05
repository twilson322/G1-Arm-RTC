#include "g1rtc/stack.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>

namespace g1rtc {

// ------------------------------------------------------------------ TaskMailbox
std::uint64_t TaskMailbox::post(std::shared_ptr<const Task> task) {
  std::lock_guard<std::mutex> lock(owners_mutex_);
  const std::uint64_t seq = ++next_seq_;
  published_.writeBuffer() = Slot{task.get(), seq};
  owners_.emplace_back(seq, std::move(task));
  published_.publish();
  return seq;
}

const Task* TaskMailbox::newest(std::uint64_t& seq) {
  if (published_.update()) {
    newest_seen_ = published_.read();
    newest_seen_seq_.store(newest_seen_.seq, std::memory_order_release);
  }
  seq = newest_seen_.seq;
  return newest_seen_.task;
}

std::size_t TaskMailbox::collect() {
  std::lock_guard<std::mutex> lock(owners_mutex_);
  const std::uint64_t seen = newest_seen_seq_.load(std::memory_order_acquire);
  const std::uint64_t executing = executing_seq_.load(std::memory_order_acquire);
  // The control thread can hold at most the task it executes and the newest one it has seen.
  // Anything posted after `seen` it has not seen yet, so that is kept too.
  const std::uint64_t oldest_in_use = executing != 0 ? std::min(executing, seen) : seen;
  std::size_t freed = 0;
  while (!owners_.empty() && owners_.front().first < oldest_in_use) {
    owners_.pop_front();
    ++freed;
  }
  return freed;
}

std::size_t TaskMailbox::retained() const {
  std::lock_guard<std::mutex> lock(owners_mutex_);
  return owners_.size();
}

// ------------------------------------------------------------------ configuration
StackConfig StackConfig::defaultsG1() {
  StackConfig c;
  c.observer.gain_per_s = VecN::Constant(10, 30.0);
  // Thresholds from the calibration in the collision experiment (1.5x the largest unexplained
  // residual over four collision-free motions); results/collision_thresholds.csv.
  c.detector.threshold_nm.resize(10);
  c.detector.threshold_nm << 0.81, 0.81, 0.82, 0.62, 0.43, 0.36, 0.41, 0.21, 0.24, 0.18;
  // order: waist yaw/roll/pitch, shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
  c.reflex_damping_nms_per_rad.resize(10);
  c.reflex_damping_nms_per_rad << 8.0, 8.0, 8.0, 2.0, 2.0, 1.0, 1.0, 0.3, 0.15, 0.15;
  return c;
}

// ------------------------------------------------------------------ ControlStack
ControlStack::ControlStack(const RobotDescription& robot, StackConfig cfg)
    : robot_(robot),
      model_(robot_.model),
      nominal_model_(robot.model),
      cfg_(std::move(cfg)),
      dyn_(model_),
      observer_(model_, cfg_.observer),
      detector_(cfg_.detector),
      wrench_(cfg_.tool_mass_kg, robot.model.gravity),
      payload_(cfg_.payload),
      supervisor_(model_, cfg_.supervisor),
      hold_(cfg_.hold),
      computed_torque_(cfg_.computed_torque_bandwidth_hz, cfg_.computed_torque_damping_ratio),
      operational_space_(cfg_.task_gains),
      impedance_(cfg_.impedance, cfg_.task_gains),
      hybrid_(cfg_.hybrid, cfg_.task_gains) {
  const int n = model_.nv;
  if (cfg_.detector.threshold_nm.size() != n || cfg_.reflex_damping_nms_per_rad.size() != n)
    throw std::invalid_argument("StackConfig: per-joint vectors must have one entry per DOF");
  q_ = qd_ = previous_impulse_ = mean_applied_torque_ = unexplained_residual_ = contact_torque_lagged_ = VecN::Zero(n);
  if (cfg_.contact_candidates_proximal_to_distal.size() > kMaxContactCandidates)
    throw std::invalid_argument("StackConfig: more contact candidates than isolateContactBody supports");
  for (const auto& name : cfg_.contact_candidates_proximal_to_distal) contact_candidates_.push_back(model_.bodyId(name));  // throws if unknown
  reference_ = Reference::hold(q_);
}

TorqueCommand ControlStack::tick(const RobotState& state) {
  const Mode mode_before = supervisor_.mode();
  updateSensing(state, mode_before);
  const bool collision = monitorCollisions(mode_before);
  const bool ready_to_start = pendingTaskStartsHere(mode_before);
  const bool task_finished = mode_before == Mode::Executing && active_task_ && time_s_ - task_start_s_ >= active_task_->duration();

  const Mode mode = supervisor_.step({time_s_, q_, qd_, state.drive_fault, collision, task_finished, ready_to_start});
  switchTask(mode_before, mode);

  VecN tau = computeTorque(mode);
  saturated_joints_ = saturate(model_, tau);
  if (cfg_.estimate_payload) updatePayloadEstimate(mode);
  writeLogRecord(mode, tau);

  TorqueCommand command;
  command.tau = tau;
  command.reset_fault = supervisor_.wantsDriveReset();
  return command;
}

void ControlStack::updateSensing(const RobotState& state, Mode mode_before) {
  // The period comes from the state timestamps, not the nominal rate: in the threaded runtime the
  // control thread's wake-ups jitter, and integrating the observer with a fixed 1 ms then made the
  // residual large enough to trip the collision detector on its own.
  const bool new_sample = !have_previous_state_ || state.time_s > time_s_;
  period_s_ = have_previous_state_ ? state.time_s - time_s_ : cfg_.nominal_period_s;
  if (have_previous_state_ && period_s_ > 0.0)
    mean_applied_torque_ = (state.applied_torque_impulse - previous_impulse_) / period_s_;
  previous_impulse_ = state.applied_torque_impulse;
  time_s_ = state.time_s;
  have_previous_state_ = true;
  q_ = state.q;
  qd_ = state.qd;

  dyn_.update(model_, robot_.tool, q_, qd_);
  Vec3 ft_position_unused;
  rbd::framePose(dyn_.kin, robot_.ft_sensor, ft_position_unused, ft_sensor_rotation_);
  if (mode_before == Mode::Enabling)
    wrench_.accumulateTare(state.wrist_ft_reading, ft_sensor_rotation_);  // arm at rest, tool in free space
  else
    wrench_.clearTare();
  tool_contact_force_ = wrench_.contactForceWorld(state.wrist_ft_reading, ft_sensor_rotation_);

  // Integrate the observer only while the drives apply our torques; restart it otherwise.
  const bool torque_controlled = mode_before == Mode::Hold || mode_before == Mode::Executing || mode_before == Mode::Reflex;
  if (!torque_controlled || !observer_running_) {
    observer_.reset(q_, qd_);
    observer_running_ = torque_controlled;
    observer_started_s_ = time_s_;
  } else if (new_sample) {
    observer_.update(q_, qd_, mean_applied_torque_, period_s_);
  }
}

bool ControlStack::monitorCollisions(Mode mode_before) {
  collision_this_tick_ = false;
  // Contact-aware detection: in a contact task the joint torque explained by the measured tool
  // force (the intended contact) is removed, so a hit on the arm is still caught while wiping. The
  // residual sees external torque through the observer's lag r = K / (s + K) tau_ext, so J^T f is
  // passed through the same lag first; an unmatched 30 Hz filter let the touchdown transient trip
  // the detector. Outside contact tasks nothing is subtracted, because a tool force there is
  // either an unexpected contact or the tool's own inertia (subtracting the latter inflated the
  // residual on fast moves).
  const bool contact_task = mode_before == Mode::Executing && active_task_ && active_task_->mode() == TrackingMode::HybridForce;
  if (contact_task && period_s_ > 0.0) {
    const VecN contact_torque = dyn_.J.topRows<3>().transpose() * tool_contact_force_;
    for (int i = 0; i < model_.nv; ++i)
      contact_torque_lagged_[i] += cfg_.observer.gain_per_s[i] * period_s_ * (contact_torque[i] - contact_torque_lagged_[i]);
  } else if (!contact_task) {
    contact_torque_lagged_.setZero();
  }
  unexplained_residual_ = observer_.residual() - contact_torque_lagged_;

  const bool monitored = mode_before == Mode::Hold || mode_before == Mode::Executing;
  if (!cfg_.detect_collisions || !monitored || !observer_running_) return false;
  if (detector_.tripped()) {
    const bool quiet = (unexplained_residual_.cwiseAbs().array() < cfg_.rearm_fraction * cfg_.detector.threshold_nm.array()).all();
    if (!quiet || mode_before != Mode::Hold) quiet_since_s_ = time_s_;
    if (time_s_ - quiet_since_s_ > cfg_.rearm_quiet_s) detector_.rearm();
    return false;
  }
  if (!detector_.update(unexplained_residual_)) return false;
  collision_this_tick_ = true;
  quiet_since_s_ = time_s_;
  isolated_body_ = isolateContactBody(model_, dyn_.kin, unexplained_residual_, cfg_.detector.threshold_nm, contact_candidates_);
  return true;
}

bool ControlStack::pendingTaskStartsHere(Mode mode_before) {
  std::uint64_t seq = 0;
  const Task* newest = mailbox_.newest(seq);
  if (seq != pending_seq_) {
    pending_task_ = newest;
    pending_seq_ = seq;
  }
  if (!pending_task_ || mode_before != Mode::Hold) return false;
  Reference first = reference_;
  pending_task_->reference(0.0, first);
  const TrackingMode tracking = pending_task_->mode();
  if (tracking == TrackingMode::ComputedTorque || tracking == TrackingMode::JointHold)
    return (first.q - q_).cwiseAbs().maxCoeff() < cfg_.start_tolerance_joint_rad;
  Vec3 position_error = first.tool_position - dyn_.tool_position;
  // a force task approaches along its push direction until contact, so it may start off the surface
  if (tracking == TrackingMode::HybridForce) position_error -= first.push_direction * first.push_direction.dot(position_error);
  return position_error.norm() < cfg_.start_tolerance_position_m &&
         rotationError(first.tool_rotation, dyn_.tool_rotation).norm() < cfg_.start_tolerance_orientation_rad;
}

void ControlStack::switchTask(Mode before, Mode after) {
  if (after == Mode::Executing && before != Mode::Executing) {
    active_task_ = pending_task_;
    active_seq_ = pending_seq_;
    task_start_s_ = time_s_;
    hybrid_.reset();
  } else if (after != Mode::Executing && before == Mode::Executing) {
    active_task_ = nullptr;
    active_seq_ = 0;
  }
  mailbox_.reportExecuting(active_seq_);
  task_time_s_ = active_task_ ? time_s_ - task_start_s_ : 0.0;
}

Controller& ControlStack::controllerFor(TrackingMode mode) {
  switch (mode) {
    case TrackingMode::JointHold: return hold_;
    case TrackingMode::ComputedTorque: return computed_torque_;
    case TrackingMode::OperationalSpace: return operational_space_;
    case TrackingMode::Impedance: return impedance_;
    case TrackingMode::HybridForce: return hybrid_;
  }
  return hold_;
}

VecN ControlStack::computeTorque(Mode mode) {
  const ControlContext context{model_, dyn_, q_, qd_, tool_contact_force_, period_s_ > 0.0 ? period_s_ : cfg_.nominal_period_s};
  const VecN damping = cfg_.reflex_damping_nms_per_rad.cwiseProduct(qd_);
  switch (mode) {
    case Mode::Disabled:
      return VecN::Zero(model_.nv);
    case Mode::Enabling:
      return dyn_.g;
    case Mode::Hold:
      reference_ = Reference::hold(supervisor_.holdPose());
      return hold_.compute(context, reference_);
    case Mode::Executing:
      active_task_->reference(task_time_s_, reference_);
      return controllerFor(active_task_->mode()).compute(context, reference_);
    case Mode::Reflex:
      switch (supervisor_.config().reflex) {
        case ReflexKind::Stop:
          reference_ = Reference::hold(supervisor_.holdPose());
          return hold_.compute(context, reference_);
        case ReflexKind::Float:
          return VecN(dyn_.g - damping);
        case ReflexKind::Retract:  // De Luca's reflex: let the residual push the arm away
          return VecN(dyn_.g + cfg_.reflex_residual_gain * observer_.residual() - damping);
        case ReflexKind::Ignore:
          break;
      }
      return dyn_.g;
    case Mode::SafeStop:
      return VecN(dyn_.g - damping);
  }
  return VecN::Zero(model_.nv);
}

void ControlStack::updatePayloadEstimate(Mode mode) {
  const bool settled = time_s_ - observer_started_s_ > cfg_.observer_settle_s;
  if (mode != Mode::Executing || !observer_running_ || !settled) return;
  payload_.update(dyn_, observer_.residual(), model_.gravity, payload_in_model_);
  if (!cfg_.adapt_model_to_payload || time_s_ - last_model_update_s_ < cfg_.payload_model_update_period_s) return;
  last_model_update_s_ = time_s_;
  const double mass = std::clamp(payload_.massKg(), 0.0, cfg_.max_payload_kg);
  const Vec3 com = payload_.comToolFrame();
  const auto tool_body = static_cast<std::size_t>(robot_.tool.body);
  model_.bodies[tool_body] = nominal_model_.bodies[tool_body];  // same size: no allocation
  model_.addPayload(robot_.tool.body, mass, com);
  payload_in_model_ << mass, mass * com;
  observer_.reset(q_, qd_);  // p = M qd just changed meaning
  observer_started_s_ = time_s_;
}

void ControlStack::writeLogRecord(Mode mode, const VecN& tau) {
  record_.time_s = time_s_;
  record_.mode = static_cast<std::uint8_t>(mode);
  record_.saturated_joints = static_cast<std::uint8_t>(saturated_joints_);
  record_.tracking_mode = active_task_ ? static_cast<std::uint8_t>(active_task_->mode()) : 0;
  const VecN& q_ref = mode == Mode::Executing ? reference_.q : supervisor_.holdPose();
  for (int i = 0; i < model_.nv; ++i) {
    record_.q[i] = static_cast<float>(q_[i]);
    record_.qd[i] = static_cast<float>(qd_[i]);
    record_.q_ref[i] = static_cast<float>(q_ref[i]);
    record_.tau[i] = static_cast<float>(tau[i]);
    record_.residual[i] = static_cast<float>(observer_.residual()[i]);
  }
  for (int k = 0; k < 3; ++k) {
    record_.tool_position[k] = static_cast<float>(dyn_.tool_position[k]);
    record_.tool_position_ref[k] = static_cast<float>(reference_.tool_position[k]);
    record_.tool_contact_force[k] = static_cast<float>(tool_contact_force_[k]);
  }
  const Eigen::Quaterniond tool_orientation(dyn_.tool_rotation);
  record_.tool_orientation_wxyz[0] = static_cast<float>(tool_orientation.w());
  record_.tool_orientation_wxyz[1] = static_cast<float>(tool_orientation.x());
  record_.tool_orientation_wxyz[2] = static_cast<float>(tool_orientation.y());
  record_.tool_orientation_wxyz[3] = static_cast<float>(tool_orientation.z());
  record_.push_force_filtered_n = static_cast<float>(hybrid_.filteredPushForce());
  record_.push_force_ref_n = static_cast<float>(reference_.push_force_n);
  record_.payload_estimate_kg = static_cast<float>(payload_.massKg());
  if (log_queue_) log_queue_->push(record_);
}

}  // namespace g1rtc
