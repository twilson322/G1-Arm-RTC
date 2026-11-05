// Threaded real-time runtime: the same RobotSim and ControlStack as the lockstep harness, on
// wall-clock threads with the plumbing a real robot needs.
//
//   robot thread    2 kHz  steps the simulator (standing in for the EtherCAT cycle), takes the
//                          newest torque command, publishes the newest sensor state
//   control thread  1 kHz  ControlStack::tick on the newest state, publishes the command
//   logger thread          drains LogRecords from a lock-free queue to CSV
//
// The robot and control threads exchange data only through triple buffers, so neither waits for
// the other: a late control thread shows up as an old command, which the drive watchdog catches,
// not as a stalled robot. Non-RT code (main, planner, ROS callbacks) talks to the stack only
// through TaskMailbox and the supervisor's atomic requests, and reads telemetry from a separate
// triple buffer.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "g1rtc/robot_sim.hpp"
#include "g1rtc/rt.hpp"
#include "g1rtc/stack.hpp"

namespace g1rtc {

struct RuntimeConfig {
  bool try_realtime = true;     // SCHED_FIFO + mlockall where permitted
  bool flush_denormals = true;  // FTZ/DAZ on the control thread (see rt::enableFlushToZero)
  int control_priority = 80, robot_priority = 85;
  int control_cpu = -1, robot_cpu = -1;  // pin the thread to this CPU if >= 0
  std::string log_path;                  // empty: no CSV log
  int log_every_nth_record = 1;
};

struct RuntimeStats {
  bool control_fifo = false, robot_fifo = false;
  std::uint64_t control_ticks = 0, robot_steps = 0;
  std::uint64_t control_skipped_periods = 0, robot_skipped_periods = 0;
  std::uint64_t control_deadline_misses = 0;  // wake-up lateness + compute > period
  std::uint64_t log_dropped = 0, log_written = 0;
  /// Robot steps that found no new command. The robot runs at twice the control rate, so about
  /// half of all steps is normal; anything above that is control-thread lateness.
  std::uint64_t robot_steps_without_new_command = 0;
  rt::LatencyHistogram control_wake_us{2.0, 5000};
  rt::LatencyHistogram control_compute_us{1.0, 2000};
  rt::LatencyHistogram robot_wake_us{2.0, 5000};
};

class Runtime {
 public:
  Runtime(RobotSim& sim, ControlStack& stack, RuntimeConfig cfg);
  ~Runtime();  // stops and joins, so a Runtime never outlives its threads
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  void start();
  void stop();
  bool running() const { return running_.load(); }
  /// Fault injection: the control thread sleeps this long once, as if blocked on I/O.
  void injectControlStall(int milliseconds) { stall_ms_.store(milliseconds); }
  /// Latest telemetry for one non-RT reader (the ROS publisher, the scenario script). Returns
  /// false if nothing new arrived since the last call.
  bool pollTelemetry(LogRecord& out);
  /// Only valid after stop(): the threads write these without synchronization while running.
  const RuntimeStats& stats() const { return stats_; }

 private:
  void robotLoop();
  void controlLoop();
  void loggerLoop();

  RobotSim& sim_;
  ControlStack& stack_;
  RuntimeConfig cfg_;
  RuntimeStats stats_;
  std::atomic<bool> running_{false};
  std::atomic<int> stall_ms_{0};
  rt::TripleBuffer<RobotState> state_buffer_;
  rt::TripleBuffer<TorqueCommand> command_buffer_;
  rt::TripleBuffer<LogRecord> telemetry_;
  std::unique_ptr<LogQueue> log_queue_;
  std::thread robot_thread_, control_thread_, logger_thread_;
};

}  // namespace g1rtc
