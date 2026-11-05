// What sets the momentum observer's residual during collision-free motion? The deployed
// configuration, then one change at a time. The collision thresholds have to sit above this floor.
#include <cmath>
#include <cstdio>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSeed = 1;

struct FloorCase {
  std::string name;
  std::string motion = "figure8";  // or "shelf"
  double encoder_bits = 18;
  double velocity_filter_hz = 150;
  double observer_gain_per_s = 30;
  bool plant_friction = true;
  bool friction_model = true;
  bool filter_matching = true;
};

std::vector<FloorCase> cases() {
  std::vector<FloorCase> out;
  for (std::string motion : {"figure8", "shelf"}) {
    out.push_back({.name = "deployed", .motion = motion});
    out.push_back({.name = "no_filter_matching", .motion = motion, .filter_matching = false});
    out.push_back({.name = "no_friction_model", .motion = motion, .friction_model = false});
    out.push_back({.name = "no_friction_model_no_matching", .motion = motion, .friction_model = false, .filter_matching = false});
  }
  out.push_back({.name = "encoder_16bit", .encoder_bits = 16});
  out.push_back({.name = "encoder_20bit", .encoder_bits = 20});
  out.push_back({.name = "gain_15", .observer_gain_per_s = 15});
  out.push_back({.name = "gain_60", .observer_gain_per_s = 60});
  out.push_back({.name = "velocity_filter_50hz", .velocity_filter_hz = 50});
  out.push_back({.name = "velocity_filter_300hz", .velocity_filter_hz = 300});
  // Near-ideal sensing (60-bit encoder, no velocity filter). Without plant friction the residual
  // should be numerical noise, which checks the observer's dynamics against MuJoCo's; with it,
  // what remains is the friction model's error alone.
  out.push_back({.name = "ideal_sensing_no_friction", .encoder_bits = 60, .velocity_filter_hz = 0, .plant_friction = false,
                 .friction_model = false});
  out.push_back({.name = "ideal_sensing", .encoder_bits = 60, .velocity_filter_hz = 0});
  return out;
}

void observerFloor() {
  std::printf("== observer residual floor (collision-free motion, max |r| per joint group, N m)\n");
  Csv csv("observer_floor.csv",
          "case,motion,encoder_bits,velocity_filter_hz,observer_gain_per_s,plant_friction,friction_model,filter_matching,"
          "max_r_waist,max_r_shoulder,max_r_elbow,max_r_wrist,rms_r_all");
  const World planning_world(SimConfig{}, kSeed, RobotDescription::homePosture());
  const PlannedMotion shelf = shelfMotion(planning_world, false, fastPlannerOptions());
  if (!shelf.ok()) throw std::runtime_error("shelf motion planning failed");

  for (const FloorCase& c : cases()) {
    SimConfig sim_config;
    sim_config.encoder_counts_per_rev = std::pow(2.0, c.encoder_bits);
    sim_config.velocity_filter_cutoff_hz = c.velocity_filter_hz;  // 0: no filter
    sim_config.joint_friction = c.plant_friction;
    StackConfig config = StackConfig::defaultsG1();
    config.detect_collisions = false;
    config.observer.gain_per_s = VecN::Constant(10, c.observer_gain_per_s);
    config.observer.model_joint_friction = c.friction_model;
    config.observer.drive_velocity_filter_hz = c.filter_matching ? c.velocity_filter_hz : 0.0;

    const bool on_shelf = c.motion == "shelf";
    Rig rig(sim_config, kSeed, on_shelf ? shelf.trajectory.q.front() : RobotDescription::homePosture(), config);
    rig.lockstep->enable(0.5);
    std::shared_ptr<const Task> task;
    if (on_shelf)
      task = std::make_shared<JointTrajectoryTask>(shelf.trajectory, TrackingMode::ComputedTorque);
    else
      task = std::make_shared<Figure8Task>(figure8At(rig.world.robot, rig.sim().trueQ()));

    VecN max_abs = VecN::Zero(10);
    double sum_sq = 0, samples = 0;
    const ControlStack& stack = *rig.stack;
    if (!rig.lockstep->execute(task, 30.0, [&](const RobotState&, const TorqueCommand&) {
          if (stack.supervisor().mode() != Mode::Executing) return;
          max_abs = max_abs.cwiseMax(stack.residual().cwiseAbs());
          sum_sq += stack.residual().squaredNorm();
          samples += 1;
        }))
      throw std::runtime_error("observer floor case did not start: " + c.name);
    // joint order: waist yaw/roll/pitch, shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
    const double waist = max_abs.head<3>().maxCoeff(), shoulder = max_abs.segment<3>(3).maxCoeff(), elbow = max_abs[6],
                 wrist = max_abs.tail<3>().maxCoeff();
    csv.row(c.name, c.motion, c.encoder_bits, c.velocity_filter_hz, c.observer_gain_per_s, c.plant_friction, c.friction_model,
            c.filter_matching, waist, shoulder, elbow, wrist, std::sqrt(sum_sq / (samples * 10.0)));
    std::printf("  %-30s %-7s waist %.3f  shoulder %.3f  elbow %.3f  wrist %.3f\n", c.name.c_str(), c.motion.c_str(), waist, shoulder,
                elbow, wrist);
  }
}

const Register kObserverFloor("observer_floor", observerFloor);

}  // namespace
}  // namespace g1rtc::experiments
