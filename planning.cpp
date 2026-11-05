// Planning from under a shelf to above it: RRT-Connect statistics over seeds, the spline, a TOPP-RA
// discretization study, and kinematic vs torque-limited vs uniform time scaling executed on the
// simulated robot with computed torque.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSceneSeed = 5, kExecutionSeed = 9;
constexpr int kRrtSeeds = 20;
constexpr std::size_t kProfileEveryNthSample = 5;  // keeps the profile CSVs small

/// Peak over the trajectory of max_j |tau_j| / tau_max_j, with tau from the controller's model.
/// If `profile` is given, writes that ratio (and which joint sets it) along the trajectory.
double plannedPeakTorqueRatio(const rbd::Model& m, const Trajectory& trajectory, Csv* profile = nullptr) {
  rbd::Kinematics<double> kin(m);
  rbd::Workspace<double> work(m);
  VecN tau(m.nv);
  double peak = 0.0;
  for (std::size_t i = 0; i < trajectory.time_s.size(); ++i) {
    rbd::forwardKinematics(m, trajectory.q[i], kin);
    rbd::rnea(m, kin, trajectory.qd[i], trajectory.qdd[i], true, work, tau);
    Eigen::Index worst_joint = 0;
    const double ratio = tau.cwiseAbs().cwiseQuotient(m.tau_max).maxCoeff(&worst_joint);
    peak = std::max(peak, ratio);
    if (profile && i % kProfileEveryNthSample == 0) profile->row(trajectory.time_s[i], trajectory.qd[i].norm(), ratio, worst_joint);
  }
  return peak;
}

struct ExecutionResult {
  double peak_torque_ratio = 0, rms_joint_error_rad = 0, max_joint_error_rad = 0, shelf_contact_ms = 0;
  long saturated_ticks = 0;
  std::string unexpected_transitions;
};

ExecutionResult executeWithComputedTorque(const Trajectory& trajectory) {
  StackConfig config = StackConfig::defaultsG1();
  config.detect_collisions = false;  // measure tracking and contact, do not stop on them
  Rig rig(SimConfig{}, kExecutionSeed, trajectory.q.front(), config);
  rig.lockstep->enable(0.5);
  const auto task = std::make_shared<JointTrajectoryTask>(trajectory, TrackingMode::ComputedTorque);
  const ControlStack& stack = *rig.stack;
  const rbd::Model& model = stack.model();
  ExecutionResult result;
  double sum_sq = 0, samples = 0;
  VecN q_ref, qd_ref, qdd_ref;
  rig.lockstep->execute(task, 30.0, [&](const RobotState&, const TorqueCommand& command) {
    if (stack.supervisor().mode() != Mode::Executing) return;
    trajectory.sample(stack.taskTime(), q_ref, qd_ref, qdd_ref);
    const VecN error = rig.sim().trueQ() - q_ref;
    sum_sq += error.squaredNorm() / model.nv;
    result.max_joint_error_rad = std::max(result.max_joint_error_rad, error.cwiseAbs().maxCoeff());
    result.peak_torque_ratio = std::max(result.peak_torque_ratio, command.tau.cwiseAbs().cwiseQuotient(model.tau_max).maxCoeff());
    result.saturated_ticks += stack.saturatedJoints() > 0;
    result.shelf_contact_ms += rig.sim().contactForceWith("shelf_board") > 0.0 ? 1e3 * Lockstep::kControlPeriodS : 0.0;
    samples += 1;
  });
  result.rms_joint_error_rad = samples > 0 ? std::sqrt(sum_sq / samples) : NAN;
  for (std::size_t i = 0; i < stack.supervisor().retainedTransitions(); ++i) {
    const Transition& t = stack.supervisor().transition(i);
    const bool expected = t.why == Reason::TaskDone || t.why == Reason::StartRequest || t.why == Reason::EnableRequest ||
                          t.why == Reason::DriveEnabled;
    if (!expected) result.unexpected_transitions += std::string(" ") + toString(t.why);
  }
  return result;
}

double milliseconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

void planning() {
  std::printf("== planning around the shelf\n");
  const World world(SimConfig{}, kSceneSeed, RobotDescription::homePosture());
  const rbd::Model& model = world.robot.model;
  const MotionPlannerOptions options = fastPlannerOptions();
  Vec3 home_position;
  Mat3 home_rotation;
  toolPose(world.robot, RobotDescription::homePosture(), home_position, home_rotation);
  const Vec3 under_shelf(0.30, -0.28, 0.90), above_shelf(0.30, -0.28, 1.10);
  const IkResult start = solveIk(model, world.robot.tool, under_shelf, home_rotation, RobotDescription::homePosture(),
                                 RobotDescription::homePosture());
  const IkResult goal = solveIk(model, world.robot.tool, above_shelf, home_rotation, start.q, RobotDescription::homePosture());
  const MotionPlanner planner(world.sim->mjModel(), world.robot, options);
  StateValidityChecker checker = planner.makeChecker();
  std::printf("  allowed-collision matrix: %d robot body pairs ignored\n", checker.allowedCollisions().allowedPairCount());
  std::printf("  IK under shelf: converged %d (%.1e m)  above: %d (%.1e m)  valid: %d %d  straight segment valid: %d\n",
              start.converged, start.position_error_m, goal.converged, goal.position_error_m, checker.isValid(start.q),
              checker.isValid(goal.q), checker.segmentValid(start.q, goal.q, options.edge_resolution_rad));
  if (!start.converged || !goal.converged) throw std::runtime_error("shelf IK failed");

  const VecN q_lower = model.q_min.array() + options.joint_limit_margin_rad;
  const VecN q_upper = model.q_max.array() - options.joint_limit_margin_rad;
  Csv rrt_csv("planning_rrt.csv", "seed,success,iterations,nodes,validity_checks,plan_ms,raw_length_rad,shortcut_length_rad,waypoints");
  std::vector<std::vector<VecN>> paths;
  std::vector<double> plan_ms;
  for (std::uint64_t seed = 1; seed <= kRrtSeeds; ++seed) {
    RrtOptions rrt = options.rrt;
    rrt.seed = seed;
    RrtStats stats;
    const auto t0 = std::chrono::steady_clock::now();
    const auto raw = rrtConnect(checker, start.q, goal.q, q_lower, q_upper, rrt, &stats);
    if (!raw) {
      rrt_csv.row(seed, 0, stats.iterations, stats.nodes, stats.validity_checks, milliseconds(t0), 0, 0, 0);
      continue;
    }
    std::mt19937_64 rng(seed);
    const std::vector<VecN> shortened = shortcut(checker, *raw, options.shortcut_attempts, options.edge_resolution_rad, rng);
    const double ms = milliseconds(t0);
    rrt_csv.row(seed, 1, stats.iterations, stats.nodes, stats.validity_checks, ms, pathLength(*raw), pathLength(shortened),
                shortened.size());
    paths.push_back(shortened);
    plan_ms.push_back(ms);
  }
  if (paths.empty()) throw std::runtime_error("RRT-Connect solved no seed");
  std::vector<double> sorted_ms = plan_ms;
  std::sort(sorted_ms.begin(), sorted_ms.end());
  std::printf("  RRT-Connect: %zu/%d solved, median plan + shortcut %.1f ms\n", paths.size(), kRrtSeeds, sorted_ms[sorted_ms.size() / 2]);

  // Time-parameterize the median-length path: representative rather than the luckiest seed.
  std::vector<std::size_t> by_length(paths.size());
  std::iota(by_length.begin(), by_length.end(), 0);
  std::sort(by_length.begin(), by_length.end(), [&](std::size_t a, std::size_t b) { return pathLength(paths[a]) < pathLength(paths[b]); });
  const auto spline = validSpline(checker, paths[by_length[by_length.size() / 2]], options.edge_resolution_rad);
  if (!spline) throw std::runtime_error("no valid spline through the median path");
  std::printf("  spline: valid, length %.3f rad\n", spline->length());
  {
    Csv path_csv("planning_path.csv", "s_rad,q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,x,y,z");
    constexpr int kSamples = 200;
    for (int k = 0; k <= kSamples; ++k) {
      const double s = spline->length() * k / kSamples;
      const VecN q = spline->configuration(s);
      Vec3 x;
      Mat3 rotation;
      toolPose(world.robot, q, x, rotation);
      path_csv.row(s, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], x.x(), x.y(), x.z());
    }
  }

  ToppLimits kinematic;
  kinematic.qd_max_rad_s = VecN::Constant(model.nv, options.max_speed_rad_s);
  kinematic.qdd_max_rad_s2 = VecN::Constant(model.nv, options.max_acceleration_rad_s2);
  ToppLimits torque = kinematic;
  torque.enforce_torque = true;
  torque.tau_max_nm = options.torque_budget_fraction * model.tau_max;

  {
    // Discretization study: grid density, and collocation vs interpolation constraints. The
    // planned peak is evaluated on the 1 ms output, so it shows what happens between grid points.
    Csv grid_csv("topp_grid.csv", "grid_points,scheme,duration_s,param_ms,planned_peak_tau_ratio");
    for (int grid : {100, 200, 400, 1000}) {
      for (bool interpolation : {false, true}) {
        ToppLimits limits = torque;
        limits.interpolation_constraints = interpolation;
        ToppStats stats;
        const auto trajectory = toppra(model, *spline, limits, grid, options.sample_period_s, &stats);
        const char* scheme = interpolation ? "interpolation" : "collocation";
        if (!trajectory) {
          std::printf("  grid %4d %-13s infeasible\n", grid, scheme);
          continue;
        }
        const double peak = plannedPeakTorqueRatio(model, *trajectory);
        grid_csv.row(grid, scheme, trajectory->duration(), 1e3 * stats.wall_time_s, peak);
        std::printf("  grid %4d %-13s duration %.3f s  param %6.1f ms  planned peak |tau|/tau_max %.2f (limit %.2f)\n", grid, scheme,
                    trajectory->duration(), 1e3 * stats.wall_time_s, peak, options.torque_budget_fraction);
      }
    }
  }

  {
    const PlannedMotion planned = planner.planToConfiguration(start.q, goal.q, 3);
    std::printf("  MotionPlanner (as deployed): %s, %s, grid %d, peak torque %.3f of the budget, duration %.3f s, %.1f ms\n",
                toString(planned.status), planned.used_rrt ? "RRT-Connect" : "straight", planned.topp_grid_points,
                planned.peak_torque_utilization, planned.trajectory.duration(), 1e3 * planned.planning_time_s);
  }

  struct Variant {
    std::string name;
    std::optional<Trajectory> trajectory;
    double param_ms;
  };
  std::vector<Variant> variants;
  ToppStats kinematic_stats, torque_stats;
  variants.push_back({"topp_kinematic", toppra(model, *spline, kinematic, options.topp_grid_points, options.sample_period_s, &kinematic_stats),
                      1e3 * kinematic_stats.wall_time_s});
  variants.push_back({"topp_torque", toppra(model, *spline, torque, options.topp_grid_points, options.sample_period_s, &torque_stats),
                      1e3 * torque_stats.wall_time_s});
  variants.push_back({"trapezoid_uniform", uniformTrapezoid(*spline, kinematic, options.sample_period_s), 0.0});

  Csv out("planning_topp.csv", "variant,feasible,duration_s,param_ms,planned_peak_tau_ratio,executed_peak_tau_ratio,"
                               "saturated_ticks,rms_joint_error_mrad,max_joint_error_mrad,shelf_contact_ms");
  for (const Variant& v : variants) {
    if (!v.trajectory) {
      out.row(v.name, 0, NAN, v.param_ms, NAN, NAN, 0, NAN, NAN, NAN);
      std::printf("  %-18s infeasible\n", v.name.c_str());
      continue;
    }
    Csv profile("topp_profile_" + v.name + ".csv", "time_s,joint_speed_norm_rad_s,tau_ratio_max,joint_of_max");
    const double planned_peak = plannedPeakTorqueRatio(model, *v.trajectory, &profile);
    const ExecutionResult e = executeWithComputedTorque(*v.trajectory);
    out.row(v.name, 1, v.trajectory->duration(), v.param_ms, planned_peak, e.peak_torque_ratio, e.saturated_ticks,
            1e3 * e.rms_joint_error_rad, 1e3 * e.max_joint_error_rad, e.shelf_contact_ms);
    std::printf("  %-18s duration %.3f s  param %.1f ms  peak |tau|/tau_max planned %.2f executed %.2f  sat %ld  rms %.2f mrad"
                "  max %.2f mrad  shelf contact %.0f ms%s%s\n",
                v.name.c_str(), v.trajectory->duration(), v.param_ms, planned_peak, e.peak_torque_ratio, e.saturated_ticks,
                1e3 * e.rms_joint_error_rad, 1e3 * e.max_joint_error_rad, e.shelf_contact_ms,
                e.unexpected_transitions.empty() ? "" : "  supervisor:", e.unexpected_transitions.c_str());
  }
}

const Register kPlanning("planning", planning);

}  // namespace
}  // namespace g1rtc::experiments
