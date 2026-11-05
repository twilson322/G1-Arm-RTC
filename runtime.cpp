#include "g1rtc/runtime.hpp"

#include <chrono>
#include <fstream>

namespace g1rtc {

Runtime::Runtime(RobotSim& sim, ControlStack& stack, RuntimeConfig cfg)
    : sim_(sim), stack_(stack), cfg_(std::move(cfg)), log_queue_(std::make_unique<LogQueue>()) {
  // Give every buffer slot its full size now, so nothing allocates once the threads run.
  state_buffer_.fill(sim_.state());
  TorqueCommand zero;
  zero.tau = VecN::Zero(sim_.nv());
  command_buffer_.fill(zero);
}

Runtime::~Runtime() { stop(); }

void Runtime::start() {
  if (running_.exchange(true)) return;
  stack_.attachLog(log_queue_.get());
  logger_thread_ = std::thread(&Runtime::loggerLoop, this);
  robot_thread_ = std::thread(&Runtime::robotLoop, this);
  control_thread_ = std::thread(&Runtime::controlLoop, this);
}

void Runtime::stop() {
  if (!running_.exchange(false)) return;
  if (control_thread_.joinable()) control_thread_.join();
  if (robot_thread_.joinable()) robot_thread_.join();
  if (logger_thread_.joinable()) logger_thread_.join();
  stack_.attachLog(nullptr);
  stats_.log_dropped = log_queue_->dropped();
}

bool Runtime::pollTelemetry(LogRecord& out) {
  if (!telemetry_.update()) return false;
  out = telemetry_.read();
  return true;
}

void Runtime::robotLoop() {
  if (cfg_.robot_cpu >= 0) rt::pinToCpu(cfg_.robot_cpu);
  stats_.robot_fifo = cfg_.try_realtime && rt::trySetRealtime(cfg_.robot_priority);
  rt::PeriodicTimer timer(std::lround(sim_.physicsDt() * 1e9));
  std::uint64_t last_seq = 0;
  while (running_.load(std::memory_order_relaxed)) {
    stats_.robot_wake_us.add(timer.wait());
    if (command_buffer_.update() && command_buffer_.read().seq != last_seq) {
      last_seq = command_buffer_.read().seq;
      sim_.command(command_buffer_.read());
    } else {
      ++stats_.robot_steps_without_new_command;
    }
    sim_.step();
    ++stats_.robot_steps;
    state_buffer_.writeBuffer() = sim_.state();
    state_buffer_.publish();
  }
  stats_.robot_skipped_periods = timer.skippedPeriods();
}

void Runtime::controlLoop() {
  if (cfg_.control_cpu >= 0) rt::pinToCpu(cfg_.control_cpu);
  stats_.control_fifo = cfg_.try_realtime && rt::trySetRealtime(cfg_.control_priority);
  if (cfg_.flush_denormals) rt::enableFlushToZero();
  constexpr long kControlPeriodNs = 1'000'000;
  rt::PeriodicTimer timer(kControlPeriodNs);
  std::uint64_t seq = 0;
  while (running_.load(std::memory_order_relaxed)) {
    const double lateness_us = timer.wait();
    stats_.control_wake_us.add(lateness_us);
    if (const int ms = stall_ms_.exchange(0); ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    const auto t0 = std::chrono::steady_clock::now();
    state_buffer_.update();
    TorqueCommand& out = command_buffer_.writeBuffer();
    out = stack_.tick(state_buffer_.read());
    out.seq = ++seq;
    command_buffer_.publish();
    const double compute_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    stats_.control_compute_us.add(compute_us);
    if (lateness_us + compute_us > kControlPeriodNs / 1000.0) ++stats_.control_deadline_misses;
    telemetry_.writeBuffer() = stack_.lastRecord();
    telemetry_.publish();
    ++stats_.control_ticks;
  }
  stats_.control_skipped_periods = timer.skippedPeriods();
}

void Runtime::loggerLoop() {
  const int nv = stack_.model().nv;
  std::ofstream file;
  if (!cfg_.log_path.empty()) {
    file.open(cfg_.log_path);
    file << "time_s,mode,saturated_joints";
    for (const char* prefix : {"q", "tau", "residual"})
      for (int i = 0; i < nv; ++i) file << ',' << prefix << i;
    file << ",x,y,z,x_ref,y_ref,z_ref\n";
  }
  LogRecord record;
  std::uint64_t index = 0;
  auto drain = [&] {
    while (log_queue_->pop(record)) {
      if (!file.is_open() || index++ % static_cast<std::uint64_t>(cfg_.log_every_nth_record) != 0) continue;
      file << record.time_s << ',' << int(record.mode) << ',' << int(record.saturated_joints);
      for (int i = 0; i < nv; ++i) file << ',' << record.q[i];
      for (int i = 0; i < nv; ++i) file << ',' << record.tau[i];
      for (int i = 0; i < nv; ++i) file << ',' << record.residual[i];
      for (float v : record.tool_position) file << ',' << v;
      for (float v : record.tool_position_ref) file << ',' << v;
      file << '\n';
      ++stats_.log_written;
    }
  };
  constexpr auto kDrainInterval = std::chrono::milliseconds(5);  // 5 records per drain at 1 kHz
  while (running_.load(std::memory_order_relaxed)) {
    drain();
    std::this_thread::sleep_for(kDrainInterval);
  }
  drain();
}

}  // namespace g1rtc
