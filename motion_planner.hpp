// The planning pipeline used by the real-time scenario, the ROS node and the experiments:
//   IK to the goal pose -> (repair an invalid start) -> straight segment if valid, else
//   RRT-Connect + shortcutting -> validated C2 spline -> TOPP-RA under velocity, acceleration and
//   torque limits -> check the sampled torque, refine the grid if needed.
// The allowed-collision matrix is sampled once at construction (about 0.1 s); each plan() builds
// its own validity checker, so concurrent plan() calls on different threads are safe.
#pragma once

#include <optional>
#include <string>

#include "g1rtc/planning.hpp"
#include "g1rtc/robot_description.hpp"

struct mjModel_;

namespace g1rtc {

struct MotionPlannerOptions {
  double clearance_m = 0.01;
  double joint_limit_margin_rad = 0.02;  // configurations closer to a limit are invalid
  double edge_resolution_rad = 0.01;
  RrtOptions rrt;
  int path_attempts = 3;  // RRT-Connect + shortcut + spline, each retry with a new seed
  int shortcut_attempts = 200;
  double max_speed_rad_s = 3.0;
  double max_acceleration_rad_s2 = 30.0;
  /// Plan to this fraction of each joint's torque limit, leaving the rest for feedback.
  double torque_budget_fraction = 0.8;
  int topp_grid_points = 400;
  /// TOPP-RA only sees the path at its grid points, and a sharp bend between two of them can push
  /// the torque past the budget there. The planner checks the sampled result and doubles the grid
  /// until the overshoot is within this fraction, up to `max_topp_grid_points`.
  double torque_overshoot_tolerance = 0.05;
  int max_topp_grid_points = 1600;
  double sample_period_s = 0.001;
  /// After a drive fault the arm sags and can come to rest inside the clearance band or a
  /// joint-limit margin, where every plan from it is rejected. Like MoveIt's
  /// FixStartStateCollision, search for a valid configuration within this inf-norm distance and
  /// prepend a slow straight move to it. That move is NOT validity-checked. 0 disables.
  double start_repair_radius_rad = 0.1;
  int start_repair_attempts = 500;
  double start_repair_speed_rad_s = 0.2, start_repair_acceleration_rad_s2 = 1.0;
};

enum class PlanStatus : std::uint8_t {
  Ok, GoalUnreachable, StartInvalid, NoPathFound, SplineInvalid, TimingInfeasible, TorqueBudgetExceeded
};
const char* toString(PlanStatus s);

struct PlannedMotion {
  PlanStatus status = PlanStatus::Ok;
  Validity start_validity = Validity::Valid;
  bool start_repaired = false;  // the trajectory begins with an unchecked move to a valid state
  bool used_rrt = false;
  int path_attempts = 0;  // RRT-Connect runs, including retries
  RrtStats rrt_stats;     // of the last run
  std::vector<VecN> waypoints;
  std::optional<JointSpline> path;
  Trajectory trajectory;
  int topp_grid_points = 0;
  double peak_torque_utilization = 0.0;  // of the torque budget, on the sampled trajectory
  double planning_time_s = 0.0;
  bool ok() const { return status == PlanStatus::Ok; }
};

class MotionPlanner {
 public:
  /// `scene` must outlive the planner; every checker it makes copies it.
  MotionPlanner(const mjModel_* scene, const RobotDescription& robot, MotionPlannerOptions options = {});
  PlannedMotion planToPose(const VecN& q_start, const Vec3& tool_position, const Mat3& tool_rotation,
                           std::uint64_t seed) const;
  PlannedMotion planToConfiguration(const VecN& q_start, const VecN& q_goal, std::uint64_t seed) const;
  StateValidityChecker makeChecker() const;
  /// A valid configuration within start_repair_radius_rad of `q`, searched from small to large
  /// perturbations (deterministic for a seed), or nullopt.
  std::optional<VecN> nearbyValidConfiguration(StateValidityChecker& checker, const VecN& q, std::uint64_t seed) const;
  const MotionPlannerOptions& options() const { return options_; }

 private:
  const mjModel_* scene_;
  RobotDescription robot_;  // a copy: the planner must not depend on the caller's lifetime
  MotionPlannerOptions options_;
  AllowedCollisionMatrix acm_;
};

}  // namespace g1rtc
