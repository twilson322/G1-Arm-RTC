// Reference generators. A Task produces a Reference for a task time and names the controller that
// should track it. Tasks are immutable after construction, so the control thread can read one
// while the planning thread builds the next (see TaskMailbox).
#pragma once

#include <memory>

#include "g1rtc/controllers.hpp"
#include "g1rtc/planning.hpp"

namespace g1rtc {

enum class TrackingMode : std::uint8_t { JointHold, ComputedTorque, OperationalSpace, Impedance, HybridForce };
const char* toString(TrackingMode m);

class Task {
 public:
  virtual ~Task() = default;
  virtual void reference(double task_time_s, Reference& r) const = 0;
  virtual double duration() const = 0;
  virtual TrackingMode mode() const = 0;
  virtual const char* name() const = 0;
};

/// Follow a time-parameterized joint trajectory (normally from the planner).
class JointTrajectoryTask final : public Task {
 public:
  explicit JointTrajectoryTask(Trajectory trajectory, TrackingMode m = TrackingMode::ComputedTorque)
      : trajectory_(std::move(trajectory)), mode_(m) {}
  void reference(double task_time_s, Reference& r) const override;
  double duration() const override { return trajectory_.duration(); }
  TrackingMode mode() const override { return mode_; }
  const char* name() const override { return "joint_trajectory"; }
  const Trajectory& trajectory() const { return trajectory_; }

 private:
  Trajectory trajectory_;
  TrackingMode mode_;
};

/// The tool traces a figure eight (Gerono lemniscate) in a plane through `center`. The amplitude
/// ramps in and out smoothly so the reference starts and ends at rest.
class Figure8Task final : public Task {
 public:
  struct Params {
    Vec3 center{0.40, -0.20, 0.95};
    Mat3 rotation = Mat3::Identity();
    Vec3 first_axis{0, 1, 0}, second_axis{0, 0, 1};  // the plane of the figure
    double first_amplitude_m = 0.10, second_amplitude_m = 0.07;
    double lap_period_s = 2.5;
    int laps = 3;
    double ramp_s = 1.0;
    VecN posture;
    TrackingMode mode = TrackingMode::OperationalSpace;
    // optional rotation of the hand about its own y axis, to excite the payload COM estimate
    double wobble_amplitude_rad = 0.0;
    double wobble_period_s = 3.7;  // deliberately not a multiple of the lap period
  };
  explicit Figure8Task(Params p) : p_(std::move(p)) {}
  void reference(double task_time_s, Reference& r) const override;
  double duration() const override { return p_.laps * p_.lap_period_s + 2 * p_.ramp_s; }
  TrackingMode mode() const override { return p_.mode; }
  const char* name() const override { return "figure8"; }
  const Params& params() const { return p_; }

 private:
  Params p_;
};

/// Wipe circles on a horizontal surface while pressing down with a set force. The circle passes
/// through `start_point` (centre at start_point - radius x); the force ramps in first. The
/// reference height is ignored by the hybrid controller, which regulates force along z instead.
class WipeTask final : public Task {
 public:
  struct Params {
    Vec3 start_point{0.42, -0.22, 0.65};
    Mat3 rotation = Mat3::Identity();
    double radius_m = 0.06;
    double lap_period_s = 3.0;
    int laps = 3;
    double push_force_n = 10.0;
    double force_ramp_s = 0.5;
    VecN posture;
  };
  explicit WipeTask(Params p) : p_(std::move(p)) {}
  void reference(double task_time_s, Reference& r) const override;
  double duration() const override;
  TrackingMode mode() const override { return TrackingMode::HybridForce; }
  const char* name() const override { return "wipe"; }
  const Params& params() const { return p_; }

 private:
  Params p_;
};

}  // namespace g1rtc
