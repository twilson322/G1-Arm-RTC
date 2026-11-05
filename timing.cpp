// Wall-clock cost of one control tick, with and without flush-to-zero. Lockstep is deterministic in
// its results, not its timing: these numbers depend on the machine and whatever else it runs.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "g1rtc/rt.hpp"
#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSeed = 4;
constexpr int kTimedTicks = 8000;  // 8 s: inside the figure-8's 9.5 s, so the task never ends

void timing() {
  std::printf("== control tick timing (lockstep, wall clock)\n");
  Csv csv("tick_timing.csv", "task,flush_to_zero,mean_us,p50_us,p99_us,max_us");
  for (bool flush_to_zero : {false, true}) {
    for (std::string which : {"hold", "operational_space", "computed_torque"}) {
      Rig rig(SimConfig{}, kSeed, RobotDescription::homePosture(), StackConfig::defaultsG1());
      rig.lockstep->enable(0.5);
      ControlStack& stack = *rig.stack;
      RobotSim& sim = rig.sim();
      const auto figure8 = std::make_shared<Figure8Task>(figure8At(rig.world.robot, sim.trueQ()));
      std::shared_ptr<const Task> task;
      if (which == "operational_space") task = figure8;
      if (which == "computed_torque")
        task = std::make_shared<JointTrajectoryTask>(jointTrajectoryFromTask(*figure8, rig.world.robot, sim.trueQ()));
      if (task) {
        stack.mailbox().post(task);
        rig.lockstep->run(2 * Lockstep::kControlPeriodS);
        stack.supervisor().requestStart();
      }
      // Time tick() alone, without the plant. FTZ/DAZ is switched on only around the tick, as on a
      // dedicated control thread, so MuJoCo's numerics are untouched.
      const int physics_steps_per_tick = static_cast<int>(std::lround(Lockstep::kControlPeriodS / sim.physicsDt()));
      const unsigned saved_float_control = rt::floatControlWord();
      std::vector<double> tick_us;
      tick_us.reserve(kTimedTicks);
      std::uint64_t seq = 0;
      for (int k = 0; k < kTimedTicks; ++k) {
        for (int i = 0; i < physics_steps_per_tick; ++i) sim.step();
        if (flush_to_zero) rt::enableFlushToZero();
        const auto t0 = std::chrono::steady_clock::now();
        TorqueCommand command = stack.tick(sim.state());
        tick_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        rt::setFloatControlWord(saved_float_control);
        command.seq = ++seq;
        sim.command(command);
      }
      if (task && stack.supervisor().mode() != Mode::Executing)
        std::printf("  warning: %s left Executing during timing\n%s", which.c_str(), transitionLog(stack.supervisor()).c_str());
      std::sort(tick_us.begin(), tick_us.end());
      const double count = static_cast<double>(tick_us.size());
      const double mean = std::accumulate(tick_us.begin(), tick_us.end(), 0.0) / count;
      const double p50 = tick_us[tick_us.size() / 2], p99 = tick_us[static_cast<std::size_t>(0.99 * count)];
      csv.row(which, flush_to_zero, mean, p50, p99, tick_us.back());
      std::printf("  %-18s FTZ %d  mean %.1f us  p50 %.1f  p99 %.1f  max %.1f\n", which.c_str(), flush_to_zero, mean, p50, p99,
                  tick_us.back());
    }
  }
}

const Register kTiming("timing", timing);

}  // namespace
}  // namespace g1rtc::experiments
