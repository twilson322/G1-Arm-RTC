// Deterministic single-threaded harness: the plant steps twice (2 kHz), then the control stack
// runs once (1 kHz), with the same command delay as the threaded runtime but no wall clock, so
// every experiment reproduces exactly from its seed.
#pragma once

#include <cmath>
#include <functional>
#include <memory>

#include "g1rtc/robot_sim.hpp"
#include "g1rtc/stack.hpp"

namespace g1rtc {

class Lockstep {
 public:
  using TickObserver = std::function<void(const RobotState&, const TorqueCommand&)>;
  static constexpr double kControlPeriodS = 0.001;

  Lockstep(RobotSim& sim, ControlStack& stack) : sim_(sim), stack_(stack) {}

  /// Run for `duration_s` of simulated time; `on_tick` sees every control tick.
  void run(double duration_s, const TickObserver& on_tick = {}) {
    const int physics_steps_per_tick = static_cast<int>(std::lround(kControlPeriodS / sim_.physicsDt()));
    const long ticks = std::lround(duration_s / kControlPeriodS);
    for (long k = 0; k < ticks; ++k) {
      for (int i = 0; i < physics_steps_per_tick; ++i) sim_.step();
      const RobotState& state = sim_.state();
      TorqueCommand command = stack_.tick(state);
      command.seq = ++command_seq_;
      sim_.command(command);
      if (on_tick) on_tick(state, command);
    }
  }

  /// Enable the drives and settle in Hold.
  void enable(double settle_s = 0.5) {
    stack_.supervisor().requestEnable();
    run(settle_s);
  }

  /// Post a task, request the start, and run until the supervisor leaves Executing or `timeout_s`
  /// passes. Returns false if the start was rejected.
  bool execute(std::shared_ptr<const Task> task, double timeout_s, const TickObserver& on_tick = {}) {
    stack_.mailbox().post(std::move(task));
    run(2 * kControlPeriodS);  // let the control thread see it
    stack_.supervisor().requestStart();
    run(2 * kControlPeriodS);
    if (stack_.supervisor().mode() != Mode::Executing) return false;
    for (double t = 0; t < timeout_s && stack_.supervisor().mode() == Mode::Executing; t += kControlPeriodS)
      run(kControlPeriodS, on_tick);
    return true;
  }

 private:
  RobotSim& sim_;
  ControlStack& stack_;
  std::uint64_t command_seq_ = 0;
};

}  // namespace g1rtc
