#include "g1rtc/motion_planner.hpp"

#include <chrono>
#include <random>

namespace g1rtc {

const char* toString(PlanStatus s) {
  switch (s) {
    case PlanStatus::Ok: return "ok";
    case PlanStatus::GoalUnreachable: return "goal_unreachable";
    case PlanStatus::StartInvalid: return "start_invalid";
    case PlanStatus::NoPathFound: return "no_path_found";
    case PlanStatus::SplineInvalid: return "spline_invalid";
    case PlanStatus::TimingInfeasible: return "timing_infeasible";
    case PlanStatus::TorqueBudgetExceeded: return "torque_budget_exceeded";
  }
  return "?";
}

MotionPlanner::MotionPlanner(const mjModel_* scene, const RobotDescription& robot, MotionPlannerOptions options)
    : scene_(scene), robot_(robot), options_(options), acm_(AllowedCollisionMatrix::sample(scene, options.clearance_m)) {}

StateValidityChecker MotionPlanner::makeChecker() const {
  return StateValidityChecker(scene_, options_.clearance_m, acm_, options_.joint_limit_margin_rad);
}

PlannedMotion MotionPlanner::planToPose(const VecN& q_start, const Vec3& tool_position, const Mat3& tool_rotation,
                                        std::uint64_t seed) const {
  const auto t0 = std::chrono::steady_clock::now();
  const IkResult ik = solveIk(robot_.model, robot_.tool, tool_position, tool_rotation, q_start, RobotDescription::homePosture());
  if (!ik.converged) {
    PlannedMotion failed;
    failed.status = PlanStatus::GoalUnreachable;
    failed.planning_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return failed;
  }
  PlannedMotion motion = planToConfiguration(q_start, ik.q, seed);
  motion.planning_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return motion;
}

std::optional<VecN> MotionPlanner::nearbyValidConfiguration(StateValidityChecker& checker, const VecN& q,
                                                            std::uint64_t seed) const {
  if (options_.start_repair_radius_rad <= 0.0) return std::nullopt;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  const rbd::Model& m = robot_.model;
  for (int attempt = 1; attempt <= options_.start_repair_attempts; ++attempt) {
    // grow the search radius linearly, so the first valid sample is close to q
    const double radius = options_.start_repair_radius_rad * attempt / options_.start_repair_attempts;
    VecN candidate = q;
    for (int j = 0; j < m.nv; ++j) candidate[j] += radius * unit(rng);
    if (checker.isValid(candidate)) return candidate;
  }
  return std::nullopt;
}

namespace {
constexpr std::uint64_t kSeedStride = 1000003;  // a prime, so retries never reuse a nearby seed

/// `second` appended to `first`; both start and end at rest and share the sample period.
Trajectory concatenate(Trajectory first, const Trajectory& second, double period_s) {
  const double offset = first.duration() + period_s;
  for (std::size_t i = 0; i < second.time_s.size(); ++i) {
    first.time_s.push_back(offset + second.time_s[i]);
    first.q.push_back(second.q[i]);
    first.qd.push_back(second.qd[i]);
    first.qdd.push_back(second.qdd[i]);
  }
  return first;
}
}  // namespace

PlannedMotion MotionPlanner::planToConfiguration(const VecN& q_start, const VecN& q_goal, std::uint64_t seed) const {
  const auto t0 = std::chrono::steady_clock::now();
  PlannedMotion motion;
  auto finish = [&](PlanStatus status) {
    motion.status = status;
    motion.planning_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return motion;
  };
  StateValidityChecker checker = makeChecker();
  motion.start_validity = checker.classify(q_start);
  VecN q_from = q_start;
  if (motion.start_validity != Validity::Valid) {
    const std::optional<VecN> repaired = nearbyValidConfiguration(checker, q_start, seed);
    if (!repaired) return finish(PlanStatus::StartInvalid);
    q_from = *repaired;
    motion.start_repaired = true;
  }
  if (!checker.isValid(q_goal)) return finish(PlanStatus::GoalUnreachable);
  if (checker.segmentValid(q_from, q_goal, options_.edge_resolution_rad)) {
    motion.waypoints = {q_from, VecN(0.5 * (q_from + q_goal)), q_goal};  // three knots: a straight spline
    motion.path = validSpline(checker, motion.waypoints, options_.edge_resolution_rad);
  } else {
    // A sampling planner's failures are seed-specific: on the shelf query about 1 seed in 40 gave
    // a waypoint sequence whose spline could not be refined into a valid one. Retry with fresh
    // seeds rather than fail the move.
    const rbd::Model& m = robot_.model;
    const VecN lower = m.q_min.array() + options_.joint_limit_margin_rad;
    const VecN upper = m.q_max.array() - options_.joint_limit_margin_rad;
    PlanStatus failure = PlanStatus::NoPathFound;
    for (int attempt = 0; attempt < options_.path_attempts && !motion.path; ++attempt) {
      const std::uint64_t attempt_seed = seed + static_cast<std::uint64_t>(attempt) * kSeedStride;
      RrtOptions rrt = options_.rrt;
      rrt.seed = attempt_seed;
      auto raw = rrtConnect(checker, q_from, q_goal, lower, upper, rrt, &motion.rrt_stats);
      motion.path_attempts = attempt + 1;
      if (!raw) {
        failure = PlanStatus::NoPathFound;
        continue;
      }
      std::mt19937_64 rng(attempt_seed);
      motion.waypoints = shortcut(checker, *raw, options_.shortcut_attempts, options_.edge_resolution_rad, rng);
      motion.used_rrt = true;
      motion.path = validSpline(checker, motion.waypoints, options_.edge_resolution_rad);
      failure = PlanStatus::SplineInvalid;
    }
    if (!motion.path) return finish(failure);
  }
  if (!motion.path) return finish(PlanStatus::SplineInvalid);
  ToppLimits limits;
  limits.qd_max_rad_s = VecN::Constant(robot_.model.nv, options_.max_speed_rad_s);
  limits.qdd_max_rad_s2 = VecN::Constant(robot_.model.nv, options_.max_acceleration_rad_s2);
  limits.enforce_torque = true;
  limits.tau_max_nm = options_.torque_budget_fraction * robot_.model.tau_max;
  for (int grid = options_.topp_grid_points; grid <= options_.max_topp_grid_points; grid *= 2) {
    auto trajectory = toppra(robot_.model, *motion.path, limits, grid, options_.sample_period_s);
    if (!trajectory) return finish(PlanStatus::TimingInfeasible);
    motion.topp_grid_points = grid;
    motion.peak_torque_utilization = peakTorqueUtilization(robot_.model, *trajectory, limits.tau_max_nm);
    motion.trajectory = std::move(*trajectory);
    if (motion.peak_torque_utilization <= 1.0 + options_.torque_overshoot_tolerance) break;
  }
  if (motion.peak_torque_utilization > 1.0 + options_.torque_overshoot_tolerance) return finish(PlanStatus::TorqueBudgetExceeded);
  if (motion.start_repaired) {
    const JointSpline escape({q_start, VecN(0.5 * (q_start + q_from)), q_from});
    ToppLimits slow;
    slow.qd_max_rad_s = VecN::Constant(robot_.model.nv, options_.start_repair_speed_rad_s);
    slow.qdd_max_rad_s2 = VecN::Constant(robot_.model.nv, options_.start_repair_acceleration_rad_s2);
    motion.trajectory = concatenate(uniformTrapezoid(escape, slow, options_.sample_period_s), motion.trajectory, options_.sample_period_s);
  }
  return finish(PlanStatus::Ok);
}

}  // namespace g1rtc
