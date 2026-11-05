#include "g1rtc/tasks.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace g1rtc {

const char* toString(TrackingMode m) {
  switch (m) {
    case TrackingMode::JointHold: return "joint_hold";
    case TrackingMode::ComputedTorque: return "computed_torque";
    case TrackingMode::OperationalSpace: return "operational_space";
    case TrackingMode::Impedance: return "impedance";
    case TrackingMode::HybridForce: return "hybrid_force";
  }
  return "?";
}

namespace {

/// A scalar signal with its first two time derivatives; products follow the product rule.
struct Signal {
  double value = 0, rate = 0, acceleration = 0;
  Signal operator*(const Signal& o) const {
    return {value * o.value, rate * o.value + value * o.rate,
            acceleration * o.value + 2 * rate * o.rate + value * o.acceleration};
  }
  Signal operator*(double k) const { return {k * value, k * rate, k * acceleration}; }
  Signal operator+(const Signal& o) const { return {value + o.value, rate + o.rate, acceleration + o.acceleration}; }
};

Signal sinusoid(double frequency_rad_s, double t) {
  const double w = frequency_rad_s;
  return {std::sin(w * t), w * std::cos(w * t), -w * w * std::sin(w * t)};
}

Signal cosine(double frequency_rad_s, double t) {
  const double w = frequency_rad_s;
  return {std::cos(w * t), -w * std::sin(w * t), -w * w * std::cos(w * t)};
}

/// Quintic smoothstep 0 -> 1 over [0, duration] with zero rate and acceleration at both ends.
Signal smoothstep(double t, double duration) {
  const double x = t / duration;
  if (x <= 0.0) return {0, 0, 0};
  if (x >= 1.0) return {1, 0, 0};
  return {x * x * x * (10 - 15 * x + 6 * x * x), 30 * x * x * (1 - x) * (1 - x) / duration,
          60 * x * (1 - x) * (1 - 2 * x) / (duration * duration)};
}

/// Rises over `ramp` after 0 and falls over `ramp` before `end`.
Signal rampInOut(double t, double end, double ramp) {
  if (t < end / 2) return smoothstep(t, ramp);
  const Signal falling = smoothstep(end - t, ramp);
  return {falling.value, -falling.rate, falling.acceleration};
}

void zeroJointMotion(const VecN& posture, Reference& r) {
  r.q = posture;
  r.qd = VecN::Zero(posture.size());
  r.qdd = VecN::Zero(posture.size());
}

}  // namespace

void JointTrajectoryTask::reference(double t, Reference& r) const { trajectory_.sample(t, r.q, r.qd, r.qdd); }

void Figure8Task::reference(double t, Reference& r) const {
  const double lap_rate = 2 * std::numbers::pi / p_.lap_period_s;
  const Signal envelope = rampInOut(t, duration(), p_.ramp_s);
  // Gerono lemniscate: first = sin(theta), second = sin(theta) cos(theta) = sin(2 theta) / 2
  const Signal first = envelope * sinusoid(lap_rate, t) * p_.first_amplitude_m;
  const Signal second = envelope * sinusoid(2 * lap_rate, t) * (0.5 * p_.second_amplitude_m);
  r.tool_position = p_.center + first.value * p_.first_axis + second.value * p_.second_axis;
  r.tool_velocity = first.rate * p_.first_axis + second.rate * p_.second_axis;
  r.tool_acceleration = first.acceleration * p_.first_axis + second.acceleration * p_.second_axis;
  // R(t) = Rot(e_y, phi) R0 with e_y the tool's y axis, so the angular velocity is e_y phi_dot
  const Signal wobble = envelope * sinusoid(2 * std::numbers::pi / p_.wobble_period_s, t) * p_.wobble_amplitude_rad;
  const Vec3 tool_y = p_.rotation.col(1);
  r.tool_rotation = Eigen::AngleAxisd(wobble.value, tool_y).toRotationMatrix() * p_.rotation;
  r.tool_angular_velocity = tool_y * wobble.rate;
  r.tool_angular_acceleration = tool_y * wobble.acceleration;
  zeroJointMotion(p_.posture, r);
}

double WipeTask::duration() const {
  constexpr double kSettleAfterLastLapS = 0.5;
  return p_.force_ramp_s + p_.laps * p_.lap_period_s + kSettleAfterLastLapS;
}

void WipeTask::reference(double t, Reference& r) const {
  const double lap_rate = 2 * std::numbers::pi / p_.lap_period_s;
  const double circling_time = p_.laps * p_.lap_period_s;
  const double tc = std::clamp(t - p_.force_ramp_s, 0.0, circling_time);
  // the radius ramps in over the first lap and out over the last, so the tangential reference
  // starts and ends at rest at the start point
  const Signal radius = rampInOut(tc, circling_time, p_.lap_period_s) * p_.radius_m;
  const Signal cos_minus_one = cosine(lap_rate, tc) + Signal{-1, 0, 0};
  const Signal x = radius * cos_minus_one;
  const Signal y = radius * sinusoid(lap_rate, tc);
  const bool circling = t > p_.force_ramp_s && t < p_.force_ramp_s + circling_time;
  r.tool_position = p_.start_point + Vec3(x.value, y.value, 0.0);
  r.tool_velocity = circling ? Vec3(x.rate, y.rate, 0.0) : Vec3::Zero();
  r.tool_acceleration = circling ? Vec3(x.acceleration, y.acceleration, 0.0) : Vec3::Zero();
  r.tool_rotation = p_.rotation;
  r.tool_angular_velocity.setZero();
  r.tool_angular_acceleration.setZero();
  r.push_direction = Vec3(0, 0, -1);
  r.push_force_n = p_.push_force_n * smoothstep(t, p_.force_ramp_s).value;
  zeroJointMotion(p_.posture, r);
}

}  // namespace g1rtc
