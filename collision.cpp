// Collision detection: calibrate per-joint thresholds on collision-free motions, check them for
// false positives on different motions, and measure detection and isolation on 48 injected hits.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

const Vec3 kWipeSurfacePoint(0.40, -0.22, 0.65);
constexpr double kWipeStartHeightM = 0.03;
constexpr double kHitDurationS = 0.15;
/// After this long past the hit onset a miss is a miss; stopping there keeps 480 runs fast.
constexpr double kDetectionWindowS = 1.0;
/// Thresholds are this multiple of the largest unexplained residual seen without collisions...
constexpr double kThresholdMargin = 1.5;
/// ...but never below this, the residual noise floor of an arm at rest.
constexpr double kMinimumThresholdNm = 0.1;

enum class Motion { Figure8Slow, Figure8Fast, Figure8Medium, Figure8Wobble, Shelf, ShelfReverse, Wipe, WipeTableHigh };

const char* toString(Motion m) {
  switch (m) {
    case Motion::Figure8Slow: return "figure8_2.5s";
    case Motion::Figure8Fast: return "figure8_1.6s";
    case Motion::Figure8Medium: return "figure8_2.0s";
    case Motion::Figure8Wobble: return "figure8_wobble";
    case Motion::Shelf: return "shelf_forward";
    case Motion::ShelfReverse: return "shelf_reverse";
    case Motion::Wipe: return "wipe";
    case Motion::WipeTableHigh: return "wipe_table+5mm";
  }
  return "?";
}

/// Trajectories and start poses that do not depend on the run's seed, computed once.
struct MotionLibrary {
  Trajectory shelf_forward, shelf_reverse;
  VecN wipe_start;
  Mat3 home_rotation;

  static MotionLibrary build() {
    const World world(SimConfig{}, 1, RobotDescription::homePosture());
    MotionLibrary lib;
    Vec3 home_position;
    toolPose(world.robot, RobotDescription::homePosture(), home_position, lib.home_rotation);
    const PlannedMotion forward = shelfMotion(world, false, fastPlannerOptions());
    const PlannedMotion reverse = shelfMotion(world, true, fastPlannerOptions());
    if (!forward.ok() || !reverse.ok()) throw std::runtime_error("shelf motion planning failed");
    lib.shelf_forward = forward.trajectory;
    lib.shelf_reverse = reverse.trajectory;
    const IkResult wipe = solveIk(world.robot.model, world.robot.tool, kWipeSurfacePoint + Vec3(0, 0, kWipeStartHeightM),
                                  lib.home_rotation, RobotDescription::homePosture(), RobotDescription::homePosture());
    if (!wipe.converged) throw std::runtime_error("wipe start pose unreachable");
    lib.wipe_start = wipe.q;
    return lib;
  }
};

struct Hit {
  std::string body;
  double force_n = 0.0;
  double onset_s = 0.0;  // task time
  Vec3 direction{1, 0, 0};
};

struct DetectionRun {
  bool ran = false;
  bool detected = false;        // after the hit onset
  bool false_positive = false;  // before the onset, or in a run without a hit
  double latency_s = NAN;
  int isolated_body = -1;
  VecN max_unexplained_before_hit;
};

DetectionRun runMotion(const StackConfig& config, Motion motion, const MotionLibrary& lib, std::uint64_t seed,
                       const Hit* hit = nullptr) {
  VecN q0 = RobotDescription::homePosture();
  if (motion == Motion::Shelf) q0 = lib.shelf_forward.q.front();
  if (motion == Motion::ShelfReverse) q0 = lib.shelf_reverse.q.front();
  if (motion == Motion::Wipe || motion == Motion::WipeTableHigh) q0 = lib.wipe_start;
  Rig rig(SimConfig{}, seed, q0, config);
  if (motion == Motion::WipeTableHigh) rig.sim().setTableHeightOffset(0.005);
  rig.lockstep->enable(0.5);

  std::shared_ptr<const Task> task;
  switch (motion) {
    case Motion::Shelf: task = std::make_shared<JointTrajectoryTask>(lib.shelf_forward); break;
    case Motion::ShelfReverse: task = std::make_shared<JointTrajectoryTask>(lib.shelf_reverse); break;
    case Motion::Wipe:
    case Motion::WipeTableHigh: {
      WipeTask::Params p;
      p.start_point = kWipeSurfacePoint;
      p.rotation = lib.home_rotation;
      p.posture = q0;
      p.force_ramp_s = 1.5;
      task = std::make_shared<WipeTask>(p);
      break;
    }
    default: {
      Figure8Task::Params p = figure8At(rig.world.robot, rig.sim().trueQ());
      p.lap_period_s = motion == Motion::Figure8Fast ? 1.6 : motion == Motion::Figure8Medium ? 2.0 : 2.5;
      if (motion == Motion::Figure8Wobble) p.wobble_amplitude_rad = 0.3;
      task = std::make_shared<Figure8Task>(p);
    }
  }

  DetectionRun run;
  run.max_unexplained_before_hit = VecN::Zero(rig.world.robot.model.nv);
  const ControlStack& stack = *rig.stack;
  const int hit_body = hit ? rig.sim().bodyId(hit->body.c_str()) : -1;
  bool hit_scheduled = false;
  double hit_onset_sim_s = INFINITY;
  const double timeout_s = hit ? hit->onset_s + kDetectionWindowS : task->duration() + 1.0;
  run.ran = rig.lockstep->execute(task, timeout_s, [&](const RobotState& state, const TorqueCommand&) {
    if (hit && !hit_scheduled) {
      hit_onset_sim_s = state.time_s - stack.taskTime() + hit->onset_s;
      rig.sim().scheduleDisturbance({hit_body, hit->force_n * hit->direction, hit_onset_sim_s, kHitDurationS});
      hit_scheduled = true;
    }
    const bool before_hit = state.time_s < hit_onset_sim_s;
    if (before_hit) run.max_unexplained_before_hit = run.max_unexplained_before_hit.cwiseMax(stack.unexplainedResidual().cwiseAbs());
    if (!stack.collisionDetectedThisTick() || run.detected || run.false_positive) return;
    if (before_hit) {
      run.false_positive = true;
    } else {
      run.detected = true;
      run.latency_s = state.time_s - hit_onset_sim_s;
      run.isolated_body = stack.isolatedBody();
    }
  });
  return run;
}

std::vector<Hit> randomHits() {
  const std::vector<std::string> struck = {"right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "tool"};
  const std::vector<double> forces_n = {5, 10, 20, 40};
  std::mt19937_64 rng(2024);
  std::uniform_real_distribution<double> unit(0, 1);
  std::vector<Hit> hits;
  for (std::size_t i = 0; i < 48; ++i) {
    Hit h;
    h.body = struck[i % struck.size()];
    h.force_n = forces_n[(i / struck.size()) % forces_n.size()];
    h.onset_s = 2.0 + 6.0 * unit(rng);
    // mostly horizontal pushes, elevation within +/-0.25 rad
    const double azimuth = 2 * std::numbers::pi * unit(rng), elevation = 0.5 * (unit(rng) - 0.5);
    h.direction = Vec3(std::cos(azimuth) * std::cos(elevation), std::sin(azimuth) * std::cos(elevation), std::sin(elevation));
    hits.push_back(h);
  }
  return hits;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return NAN;
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1))];
}

void collision() {
  std::printf("== collision detection\n");
  const MotionLibrary lib = MotionLibrary::build();
  const World reference(SimConfig{}, 1, RobotDescription::homePosture());
  const rbd::Model& model = reference.robot.model;
  const int n = model.nv;
  const std::vector<std::string>& chain = StackConfig::defaultsG1().contact_candidates_proximal_to_distal;
  // position along the chain, so "within one link" can be scored
  auto chainIndex = [&](int body) -> int {
    for (std::size_t i = 0; i < chain.size(); ++i)
      if (model.bodyId(chain[i]) == body) return static_cast<int>(i);
    return -100;
  };
  const std::vector<Hit> hits = randomHits();

  Csv calibration_csv("collision_calibration.csv", "friction_model,motion,joint,max_unexplained_residual_nm");
  Csv threshold_csv("collision_thresholds.csv", "friction_model,joint,threshold_nm");
  Csv summary_csv("collision_detection.csv",
                  "threshold_scale,friction_model,hits,detected,tripped_before_hit,false_positive_runs,validation_runs,"
                  "median_latency_ms,p90_latency_ms,isolation_exact,isolation_within_one_link");
  Csv hits_csv("collision_hits.csv", "threshold_scale,friction_model,body,force_n,detected,latency_ms,isolated_body");
  const std::vector<Motion> calibration_motions = {Motion::Figure8Slow, Motion::Figure8Fast, Motion::Shelf, Motion::Wipe};
  const std::vector<Motion> validation_motions = {Motion::Figure8Medium, Motion::Figure8Wobble, Motion::ShelfReverse,
                                                  Motion::WipeTableHigh};

  for (bool friction_model : {true, false}) {
    StackConfig calibration = StackConfig::defaultsG1();
    calibration.observer.model_joint_friction = friction_model;
    calibration.detect_collisions = false;
    VecN worst = VecN::Zero(n);
    std::uint64_t seed = 11;
    for (Motion motion : calibration_motions) {
      const DetectionRun r = runMotion(calibration, motion, lib, seed++);
      if (!r.ran) throw std::runtime_error(std::string("calibration motion did not start: ") + toString(motion));
      for (int j = 0; j < n; ++j) calibration_csv.row(friction_model, toString(motion), j, r.max_unexplained_before_hit[j]);
      worst = worst.cwiseMax(r.max_unexplained_before_hit);
    }
    const VecN thresholds = (kThresholdMargin * worst).cwiseMax(VecN::Constant(n, kMinimumThresholdNm));
    for (int j = 0; j < n; ++j) threshold_csv.row(friction_model, j, thresholds[j]);
    std::printf("  friction model %d, calibrated thresholds (N m):", friction_model);
    for (int j = 0; j < n; ++j) std::printf(" %.2f", thresholds[j]);
    std::printf("\n");
    // The deployed defaults are this calibration, copied by hand; say so loudly when they drift.
    const VecN& deployed = StackConfig::defaultsG1().detector.threshold_nm;
    constexpr double kDriftTolerance = 0.05;
    if (friction_model && ((thresholds - deployed).cwiseAbs().array() > kDriftTolerance * deployed.array()).any())
      std::printf("  WARNING: StackConfig::defaultsG1() thresholds differ from this calibration by more than %.0f %%\n",
                  100 * kDriftTolerance);

    for (double scale : {0.5, 0.75, 1.0, 1.5, 2.0}) {
      StackConfig config = StackConfig::defaultsG1();
      config.observer.model_joint_friction = friction_model;
      config.detector.threshold_nm = scale * thresholds;
      config.supervisor.reflex = ReflexKind::Ignore;  // detection only: the task keeps running
      int false_positive_runs = 0, validation_runs = 0;
      std::uint64_t validation_seed = 31;
      for (Motion motion : validation_motions) {
        const DetectionRun r = runMotion(config, motion, lib, validation_seed++);
        false_positive_runs += r.false_positive;
        validation_runs += r.ran;
      }
      std::vector<double> latencies_ms;
      // The detector fires once per run, so a run that trips before its hit cannot detect the hit.
      int detected = 0, tripped_before_hit = 0, isolated_exactly = 0, isolated_within_one = 0;
      for (std::size_t i = 0; i < hits.size(); ++i) {
        const DetectionRun r = runMotion(config, Motion::Figure8Slow, lib, 100 + i, &hits[i]);
        tripped_before_hit += r.false_positive;
        hits_csv.row(scale, friction_model, hits[i].body, hits[i].force_n, r.detected, 1e3 * r.latency_s,
                     r.isolated_body >= 0 ? model.bodies[static_cast<std::size_t>(r.isolated_body)].name : "none");
        if (!r.detected) continue;
        ++detected;
        latencies_ms.push_back(1e3 * r.latency_s);
        const int truth = chainIndex(model.bodyId(hits[i].body)), estimate = chainIndex(r.isolated_body);
        isolated_exactly += estimate == truth;
        isolated_within_one += std::abs(estimate - truth) <= 1;
      }
      const double median_ms = percentile(latencies_ms, 0.5), p90_ms = percentile(latencies_ms, 0.9);
      summary_csv.row(scale, friction_model, hits.size(), detected, tripped_before_hit, false_positive_runs, validation_runs,
                      median_ms, p90_ms, isolated_exactly, isolated_within_one);
      std::printf("  friction model %d  scale %.2f  detected %2d/%zu (tripped early %2d)  false-positive runs %d/%d"
                  "  latency median %.0f ms p90 %.0f ms  isolation exact %d/%d within one link %d/%d\n",
                  friction_model, scale, detected, hits.size(), tripped_before_hit, false_positive_runs, validation_runs, median_ms,
                  p90_ms, isolated_exactly, detected, isolated_within_one, detected);
    }
  }
}

const Register kCollision("collision", collision);

}  // namespace
}  // namespace g1rtc::experiments
