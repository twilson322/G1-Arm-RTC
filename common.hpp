#pragma once
#include <memory>
#include <string>

#include "g1rtc/lockstep.hpp"
#include "g1rtc/planning.hpp"

namespace g1test {
using namespace g1rtc;

inline const char* const kScene = G1RTC_MODEL_DIR "/g1_upper_body.xml";

inline VecN home() { return RobotDescription::homePosture(); }

/// Sensing and actuation without the realism: exact enough to test the algebra in closed loop.
inline SimConfig idealSim(bool friction = false) {
  SimConfig s;
  s.encoder_counts_per_rev = 1e12;
  s.velocity_filter_cutoff_hz = 0.0;  // no filter
  s.ft_force_noise_std_n = 0.0;
  s.ft_torque_noise_std_nm = 0.0;
  s.joint_friction = friction;
  return s;
}

struct Rig {
  std::unique_ptr<RobotSim> sim;
  RobotDescription robot;
  std::unique_ptr<ControlStack> stack;
  std::unique_ptr<Lockstep> lockstep;
  Rig(const SimConfig& sim_config, const VecN& q0, const StackConfig& config = StackConfig::defaultsG1(), std::uint64_t seed = 1)
      : sim(std::make_unique<RobotSim>(kScene, sim_config, seed)), robot(RobotDescription::fromMujoco(sim->mjModel())) {
    sim->reset(q0);
    stack = std::make_unique<ControlStack>(robot, config);
    lockstep = std::make_unique<Lockstep>(*sim, *stack);
  }
  /// Supervisor history, for failure messages.
  std::string transitions() const {
    std::string out;
    const Supervisor& sv = stack->supervisor();
    for (std::size_t i = 0; i < sv.retainedTransitions(); ++i) {
      const Transition& t = sv.transition(i);
      out += std::to_string(t.time_s) + " " + toString(t.from) + "->" + toString(t.to) + " (" + toString(t.why) + ")\n";
    }
    return out;
  }
  void toolPose(const VecN& q, Vec3& position, Mat3& rotation) const {
    rbd::DynamicsCache dyn(robot.model);
    dyn.update(robot.model, robot.tool, q, VecN::Zero(robot.model.nv));
    position = dyn.tool_position;
    rotation = dyn.tool_rotation;
  }
};
}  // namespace g1test
