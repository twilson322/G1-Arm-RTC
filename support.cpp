#include "support.hpp"

#include <mujoco/mujoco.h>

#include <cstdio>

namespace g1rtc::experiments {

std::filesystem::path& outputDirectory() {
  static std::filesystem::path dir = "results";
  return dir;
}

Csv::Csv(const std::string& file_name, const std::string& header) : file_(outputDirectory() / file_name) {
  file_ << header << '\n';
}

World::World(const SimConfig& sim_config, std::uint64_t seed, const VecN& q0)
    : sim(std::make_unique<RobotSim>(kSceneXml, sim_config, seed)), robot(RobotDescription::fromMujoco(sim->mjModel())) {
  sim->reset(q0);
}

Rig::Rig(const SimConfig& sim_config, std::uint64_t seed, const VecN& q0, const StackConfig& stack_config)
    : world(sim_config, seed, q0),
      stack(std::make_unique<ControlStack>(world.robot, stack_config)),
      lockstep(std::make_unique<Lockstep>(*world.sim, *stack)) {}

void toolPose(const RobotDescription& robot, const VecN& q, Vec3& position, Mat3& rotation) {
  rbd::Kinematics<double> kin(robot.model);
  rbd::forwardKinematics(robot.model, q, kin);
  rbd::framePose(kin, robot.tool, position, rotation);
}

Figure8Task::Params figure8At(const RobotDescription& robot, const VecN& q0) {
  Figure8Task::Params p;
  toolPose(robot, q0, p.center, p.rotation);
  p.posture = q0;
  return p;
}

Trajectory jointTrajectoryFromTask(const Task& task, const RobotDescription& robot, const VecN& q0) {
  constexpr double kPeriodS = 0.001;
  // The joint path is differentiated twice, so IK error becomes acceleration noise: solve tightly.
  IkOptions tight;
  tight.max_iterations = 200;
  tight.position_tolerance_m = 1e-10;
  tight.orientation_tolerance_rad = 1e-9;
  Trajectory out;
  VecN q = q0;
  Reference r;
  r.q = q0;
  for (double t = 0; t <= task.duration() + 1e-9; t += kPeriodS) {
    task.reference(t, r);
    q = solveIk(robot.model, robot.tool, r.tool_position, r.tool_rotation, q, q0, tight).q;
    out.time_s.push_back(t);
    out.q.push_back(q);
  }
  const std::size_t n = out.q.size();
  out.qd.assign(n, VecN::Zero(robot.model.nv));
  out.qdd.assign(n, VecN::Zero(robot.model.nv));
  for (std::size_t i = 1; i + 1 < n; ++i) {
    out.qd[i] = (out.q[i + 1] - out.q[i - 1]) / (2 * kPeriodS);
    out.qdd[i] = (out.q[i + 1] - 2 * out.q[i] + out.q[i - 1]) / (kPeriodS * kPeriodS);
  }
  return out;
}

MotionPlannerOptions fastPlannerOptions() {
  MotionPlannerOptions o;
  o.max_speed_rad_s = 6.0;
  o.max_acceleration_rad_s2 = 80.0;
  return o;
}

PlannedMotion shelfMotion(const World& world, bool reverse, const MotionPlannerOptions& options) {
  const Vec3 under_shelf(0.30, -0.28, 0.90), above_shelf(0.30, -0.28, 1.10);
  constexpr std::uint64_t kSeed = 3;
  Vec3 home_position;
  Mat3 home_rotation;
  toolPose(world.robot, RobotDescription::homePosture(), home_position, home_rotation);
  const VecN q_under = solveIk(world.robot.model, world.robot.tool, under_shelf, home_rotation, RobotDescription::homePosture(),
                               RobotDescription::homePosture()).q;
  const VecN q_above = solveIk(world.robot.model, world.robot.tool, above_shelf, home_rotation, q_under,
                               RobotDescription::homePosture()).q;
  const MotionPlanner planner(world.sim->mjModel(), world.robot, options);
  return reverse ? planner.planToConfiguration(q_above, q_under, kSeed) : planner.planToConfiguration(q_under, q_above, kSeed);
}

std::string transitionLog(const Supervisor& supervisor) {
  std::string out;
  for (std::size_t i = 0; i < supervisor.retainedTransitions(); ++i) {
    const Transition& t = supervisor.transition(i);
    char line[160];
    std::snprintf(line, sizeof(line), "    %8.3f s  %-9s -> %-9s (%s)\n", t.time_s, toString(t.from), toString(t.to), toString(t.why));
    out += line;
  }
  return out;
}

std::map<std::string, Experiment>& registry() {
  static std::map<std::string, Experiment> experiments;
  return experiments;
}

}  // namespace g1rtc::experiments
