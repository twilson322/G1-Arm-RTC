// Shared scaffolding for the deterministic experiments.
#pragma once

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "g1rtc/lockstep.hpp"
#include "g1rtc/motion_planner.hpp"

namespace g1rtc::experiments {

inline const std::string kSceneXml = G1RTC_MODEL_DIR "/g1_upper_body.xml";

/// Where CSVs go; set once by main().
std::filesystem::path& outputDirectory();

/// CSV file with a header, closed on destruction.
class Csv {
 public:
  Csv(const std::string& file_name, const std::string& header);
  template <class... Values>
  void row(const Values&... values) {
    std::ostringstream line;
    line.precision(7);
    std::size_t column = 0;
    ((line << (column++ ? "," : "") << values), ...);
    file_ << line.str() << '\n';
  }

 private:
  std::ofstream file_;
};

/// A simulated robot plus the controller-side description of it.
struct World {
  std::unique_ptr<RobotSim> sim;
  RobotDescription robot;
  World(const SimConfig& sim_config, std::uint64_t seed, const VecN& q0);
};

/// A world, its control stack and the lockstep harness, ready to enable.
struct Rig {
  World world;
  std::unique_ptr<ControlStack> stack;
  std::unique_ptr<Lockstep> lockstep;
  Rig(const SimConfig& sim_config, std::uint64_t seed, const VecN& q0, const StackConfig& stack_config);
  RobotSim& sim() { return *world.sim; }
};

void toolPose(const RobotDescription& robot, const VecN& q, Vec3& position, Mat3& rotation);
Figure8Task::Params figure8At(const RobotDescription& robot, const VecN& q0);

/// Joint-space version of a Cartesian task for the joint-space controllers: IK every millisecond
/// (warm-started, pulled toward the task's posture), then central differences.
Trajectory jointTrajectoryFromTask(const Task& task, const RobotDescription& robot, const VecN& q0);

/// The planning experiment's motion under the shelf to above it (or back), deterministic.
PlannedMotion shelfMotion(const World& world, bool reverse, const MotionPlannerOptions& options);
MotionPlannerOptions fastPlannerOptions();  // the limits the planning experiments use

/// Supervisor transitions as "t from->to (reason)" lines, for diagnostics.
std::string transitionLog(const Supervisor& supervisor);

using Experiment = std::function<void()>;
std::map<std::string, Experiment>& registry();
struct Register {
  Register(const std::string& name, Experiment fn) { registry()[name] = std::move(fn); }
};

}  // namespace g1rtc::experiments
