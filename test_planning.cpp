#include <gtest/gtest.h>

#include <random>

#include "common.hpp"
#include "g1rtc/motion_planner.hpp"

using namespace g1test;

namespace {

constexpr double kClearanceM = 0.01;
constexpr double kResolutionRad = 0.01;

std::vector<VecN> randomWaypoints(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> step(0, 0.3);
  std::vector<VecN> waypoints;
  VecN q = home();
  for (int i = 0; i < n; ++i) {
    waypoints.push_back(q);
    for (int j = 0; j < 10; ++j) q[j] += step(rng);
  }
  return waypoints;
}

StateValidityChecker makeChecker(const Rig& r) {
  return StateValidityChecker(r.sim->mjModel(), kClearanceM, AllowedCollisionMatrix::sample(r.sim->mjModel(), kClearanceM));
}

}  // namespace

TEST(Planning, SplineInterpolatesAndIsTwiceDifferentiable) {
  const auto waypoints = randomWaypoints(6, 1);
  const JointSpline spline(waypoints);
  double s = 0;
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    if (i > 0) s += (waypoints[i] - waypoints[i - 1]).norm();
    EXPECT_LT((spline.configuration(s) - waypoints[i]).norm(), 1e-9);
  }
  for (double fraction : {0.065, 0.25, 0.55, 0.85}) {
    const double h = 1e-5, at = fraction * spline.length();
    const VecN tangent_fd = (spline.configuration(at + h) - spline.configuration(at - h)) / (2 * h);
    const VecN curvature_fd = (spline.tangent(at + h) - spline.tangent(at - h)) / (2 * h);
    EXPECT_LT((spline.tangent(at) - tangent_fd).norm(), 1e-6);
    EXPECT_LT((spline.curvature(at) - curvature_fd).norm(), 1e-5);
  }
}

TEST(Planning, ToppraMatchesTrapezoidOnOneJoint) {
  // A straight path moving one joint by 1 rad: the time-optimal profile is a trapezoid.
  Rig r(idealSim(), home());
  VecN a = home(), b = home();
  b[6] += 1.0;
  const JointSpline path({a, VecN(0.5 * (a + b)), b});
  ToppLimits limits;
  limits.qd_max_rad_s = VecN::Constant(10, 2.0);
  limits.qdd_max_rad_s2 = VecN::Constant(10, 10.0);
  ToppStats stats;
  const auto tr = toppra(r.robot.model, path, limits, 500, 0.001, &stats);
  ASSERT_TRUE(tr.has_value());
  EXPECT_EQ(stats.status, ToppStatus::Ok);
  // accelerate to 2 rad/s in 0.2 s (0.2 rad), cruise 0.6 rad (0.3 s), decelerate in 0.2 s
  EXPECT_NEAR(tr->duration(), 0.7, 0.01);
  for (std::size_t i = 0; i < tr->time_s.size(); ++i) {
    EXPECT_LE(std::abs(tr->qd[i][6]), 2.0 * 1.01);
    EXPECT_LE(std::abs(tr->qdd[i][6]), 10.0 * 1.05);
  }
}

TEST(Planning, ToppraTorqueLimitsHoldBetweenGridPoints) {
  Rig r(idealSim(), home());
  const JointSpline path(randomWaypoints(4, 7));
  ToppLimits limits;
  limits.qd_max_rad_s = VecN::Constant(10, 6.0);
  limits.qdd_max_rad_s2 = VecN::Constant(10, 100.0);
  limits.enforce_torque = true;
  limits.tau_max_nm = 0.6 * r.robot.model.tau_max;
  // A coarse grid on purpose: with collocation only, the limit was exceeded 4x between points.
  const auto tr = toppra(r.robot.model, path, limits, 100, 0.001);
  ASSERT_TRUE(tr.has_value());
  const double peak = peakTorqueUtilization(r.robot.model, *tr, limits.tau_max_nm);
  EXPECT_LE(peak, 1.05);
  EXPECT_GT(peak, 0.9);  // torque is the active constraint, so the test means something
}

TEST(Planning, IkReachesReachablePoseAndRejectsUnreachable) {
  Rig r(idealSim(), home());
  Vec3 x;
  Mat3 R;
  r.toolPose(home(), x, R);
  const IkResult reachable = solveIk(r.robot.model, r.robot.tool, x + Vec3(0.05, -0.05, 0.08), R, home(), home());
  EXPECT_TRUE(reachable.converged);
  EXPECT_LT(reachable.position_error_m, 1e-4);
  const IkResult unreachable = solveIk(r.robot.model, r.robot.tool, Vec3(2.0, 0, 1.0), R, home(), home());
  EXPECT_FALSE(unreachable.converged);
}

TEST(Planning, ValidityCheckerSeesTheTableAndJointLimits) {
  Rig r(idealSim(), home());
  StateValidityChecker checker = makeChecker(r);
  EXPECT_TRUE(checker.isValid(home()));
  Vec3 x;
  Mat3 R;
  r.toolPose(home(), x, R);
  const IkResult in_table = solveIk(r.robot.model, r.robot.tool, Vec3(0.40, -0.22, 0.64), R, home(), home());
  ASSERT_TRUE(in_table.converged);
  EXPECT_FALSE(checker.isValid(in_table.q));
  // past a joint limit, in free space (an earlier version checked collisions only)
  VecN past_limit = home();
  past_limit[6] = r.robot.model.q_max[6] + 0.05;
  EXPECT_FALSE(checker.isValid(past_limit));
}

TEST(Planning, NearDuplicateWaypointsAreDroppedBeforeFitting) {
  Rig r(idealSim(), home());
  StateValidityChecker checker = makeChecker(r);
  VecN a = home(), b = home(), c = home();
  b[3] -= 0.3;
  c[3] -= 0.3;
  c[6] += 0.3;
  // What shortcutting can leave behind: a knot 3 mrad from the start, off the direction of travel.
  const VecN almost_a = a + 0.003 * VecN::Unit(10, 0);
  const std::vector<VecN> waypoints = {a, almost_a, b, c};
  const auto kept = dropCloseWaypoints(checker, waypoints, 0.05, kResolutionRad);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_EQ(kept.front(), a);
  EXPECT_EQ(kept.back(), c);
  auto peakCurvature = [](const JointSpline& spline) {
    double peak = 0;
    for (int k = 0; k <= 2000; ++k) peak = std::max(peak, spline.curvature(spline.length() * k / 2000).norm());
    return peak;
  };
  EXPECT_GT(peakCurvature(JointSpline(waypoints)), peakCurvature(JointSpline(kept)));
  // nothing to drop: the waypoints come back unchanged
  EXPECT_EQ(dropCloseWaypoints(checker, kept, 0.05, kResolutionRad).size(), kept.size());
}

TEST(Planning, RrtConnectFindsValidPathAroundShelf) {
  Rig r(idealSim(), home());
  Vec3 x;
  Mat3 R;
  r.toolPose(home(), x, R);
  const IkResult a = solveIk(r.robot.model, r.robot.tool, Vec3(0.30, -0.28, 0.90), R, home(), home());
  const IkResult b = solveIk(r.robot.model, r.robot.tool, Vec3(0.30, -0.28, 1.10), R, a.q, home());
  ASSERT_TRUE(a.converged && b.converged);
  StateValidityChecker checker = makeChecker(r);
  ASSERT_TRUE(checker.isValid(a.q) && checker.isValid(b.q));
  ASSERT_FALSE(checker.segmentValid(a.q, b.q, kResolutionRad));  // the shelf is in the way
  const VecN lower = r.robot.model.q_min.array() + 0.02, upper = r.robot.model.q_max.array() - 0.02;
  for (std::uint64_t seed : {1, 2, 3}) {
    RrtOptions options;
    options.seed = seed;
    const auto path = rrtConnect(checker, a.q, b.q, lower, upper, options);
    ASSERT_TRUE(path.has_value());
    EXPECT_LT((path->front() - a.q).norm(), 1e-12);
    EXPECT_LT((path->back() - b.q).norm(), 1e-12);
    for (std::size_t i = 1; i < path->size(); ++i) EXPECT_TRUE(checker.segmentValid((*path)[i - 1], (*path)[i], kResolutionRad));
    std::mt19937_64 rng(seed);
    const auto shortened = shortcut(checker, *path, 100, kResolutionRad, rng);
    EXPECT_LE(pathLength(shortened), pathLength(*path) + 1e-9);
    EXPECT_TRUE(validSpline(checker, shortened, kResolutionRad).has_value());
  }
}

TEST(Planning, MotionPlannerProducesExecutableTrajectory) {
  Rig r(idealSim(), home());
  const MotionPlanner planner(r.sim->mjModel(), r.robot);
  Vec3 x;
  Mat3 R;
  r.toolPose(home(), x, R);
  const PlannedMotion motion = planner.planToPose(home(), Vec3(0.30, -0.28, 1.10), R, 3);
  ASSERT_TRUE(motion.ok()) << toString(motion.status);
  const Trajectory& tr = motion.trajectory;
  EXPECT_LT((tr.q.front() - home()).norm(), 1e-9);
  EXPECT_LT(tr.qd.front().norm(), 1e-6);
  EXPECT_LT(tr.qd.back().norm(), 1e-6);
  Vec3 end;
  r.toolPose(tr.q.back(), end, R);
  EXPECT_LT((end - Vec3(0.30, -0.28, 1.10)).norm(), 1e-3);
  StateValidityChecker checker = planner.makeChecker();
  for (std::size_t i = 0; i < tr.q.size(); i += 10) EXPECT_TRUE(checker.isValid(tr.q[i])) << "sample " << i;
  const MotionPlannerOptions& o = planner.options();
  EXPECT_LE(peakTorqueUtilization(r.robot.model, tr, o.torque_budget_fraction * r.robot.model.tau_max), 1.05);

  const PlannedMotion unreachable = planner.planToPose(home(), Vec3(2.0, 0.0, 1.0), R, 3);
  EXPECT_EQ(unreachable.status, PlanStatus::GoalUnreachable);
}

TEST(Planning, InvalidStartIsRepairedWithASlowEscapeMove) {
  // An arm that sagged into the table after a drive fault: the planner must still get it out.
  Rig r(idealSim(), home());
  const MotionPlanner planner(r.sim->mjModel(), r.robot);
  Vec3 x;
  Mat3 R;
  r.toolPose(home(), x, R);
  const IkResult on_table = solveIk(r.robot.model, r.robot.tool, Vec3(0.40, -0.22, 0.655), R, home(), home());
  ASSERT_TRUE(on_table.converged);
  StateValidityChecker checker = planner.makeChecker();
  ASSERT_EQ(checker.classify(on_table.q), Validity::InCollision);  // inside the 1 cm clearance band

  const PlannedMotion motion = planner.planToConfiguration(on_table.q, home(), 7);
  ASSERT_TRUE(motion.ok()) << toString(motion.status);
  EXPECT_TRUE(motion.start_repaired);
  EXPECT_EQ(motion.start_validity, Validity::InCollision);
  const Trajectory& tr = motion.trajectory;
  EXPECT_LT((tr.q.front() - on_table.q).norm(), 1e-9);  // starts where the arm is
  EXPECT_LT((tr.q.back() - home()).norm(), 1e-9);
  for (std::size_t i = 1; i < tr.time_s.size(); ++i) EXPECT_NEAR(tr.time_s[i] - tr.time_s[i - 1], 0.001, 1e-9);

  MotionPlannerOptions no_repair;
  no_repair.start_repair_radius_rad = 0.0;
  const MotionPlanner strict(r.sim->mjModel(), r.robot, no_repair);
  EXPECT_EQ(strict.planToConfiguration(on_table.q, home(), 7).status, PlanStatus::StartInvalid);
}
