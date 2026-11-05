// Threaded real-time run of the full stack with a scripted scenario:
//   enable -> move under the shelf -> plan around the shelf to above it ->
//   injected 20 ms control-thread stall (the drive watchdog must trip) -> operator reset -> home.
// Planning runs on its own non-real-time thread and hands trajectories to the 1 kHz loop through
// the TaskMailbox. A move that ends in SafeStop or Reflex for any other reason (a latency spike
// tripping the watchdog under load, say) is reported, reset and re-planned from where the arm
// is, at most kMaxAttemptsPerMove times. The run reports wake-up latency, compute time, deadline
// misses, every unplanned fault, and the supervisor's transition log.
//
//   g1rtc_rt [--stress N] [--no-rt] [--out DIR] [--tag NAME]
#include <mujoco/mujoco.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "g1rtc/motion_planner.hpp"
#include "g1rtc/runtime.hpp"

using namespace g1rtc;
using namespace std::chrono_literals;

namespace {

constexpr int kMaxAttemptsPerMove = 3;
constexpr int kInjectedStallMs = 20;  // four times the 5 ms drive watchdog
constexpr std::uint64_t kSimSeed = 1;
constexpr int kMoves = 3;

struct Options {
  int stress_threads = 0;
  bool try_realtime = true;
  std::string out_dir = "results", tag = "nominal";
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (!std::strcmp(argv[i], "--stress") && has_value) o.stress_threads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-rt")) o.try_realtime = false;
    else if (!std::strcmp(argv[i], "--out") && has_value) o.out_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--tag") && has_value) o.tag = argv[++i];
    else {
      std::fprintf(stderr, "usage: %s [--stress N] [--no-rt] [--out DIR] [--tag NAME]\n", argv[0]);
      std::exit(2);
    }
  }
  return o;
}

/// The scenario script: everything here runs on non-RT threads and touches the stack only through
/// the mailbox, the supervisor's atomic requests and the telemetry buffer.
class Scenario {
 public:
  Scenario(Runtime& runtime, ControlStack& stack, const MotionPlanner& planner, const std::string& events_path)
      : runtime_(runtime), stack_(stack), planner_(planner), events_(events_path), t0_(std::chrono::steady_clock::now()) {}

  double wallS() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }

  template <class... Args>
  void event(const char* format, Args... args) {
    char line[256];
    std::snprintf(line, sizeof(line), format, args...);
    std::printf("[%6.3f s] %s\n", wallS(), line);
    events_ << wallS() << ' ' << line << '\n';
  }

  bool waitFor(Mode mode, double timeout_s) const {
    const double start = wallS();
    while (stack_.supervisor().mode() != mode && wallS() - start < timeout_s) std::this_thread::sleep_for(2ms);
    return stack_.supervisor().mode() == mode;
  }

  /// Get back to Hold from wherever a move ended.
  bool recover() {
    switch (stack_.supervisor().mode()) {
      case Mode::Hold: return true;                            // a reflex already finished
      case Mode::Reflex: return waitFor(Mode::Hold, 2.0);      // let it finish
      case Mode::Executing: stack_.supervisor().requestStop(); // overran its time budget
                            waitFor(Mode::SafeStop, 0.5);
                            break;
      default: break;
    }
    std::this_thread::sleep_for(500ms);  // an operator would look before resetting
    stack_.supervisor().requestReset();
    return waitFor(Mode::Hold, 2.0);
  }

  VecN measuredQ() {
    LogRecord telemetry;
    while (!runtime_.pollTelemetry(telemetry)) std::this_thread::sleep_for(1ms);
    VecN q(stack_.model().nv);
    for (int i = 0; i < q.size(); ++i) q[i] = telemetry.q[i];
    return q;
  }

  using Goal = std::variant<Vec3, VecN>;  // a tool position (orientation kept) or a configuration

  /// Plan from the current pose and execute, re-planning after an unplanned fault.
  bool move(const char* name, const Goal& goal, const Mat3& tool_rotation, std::uint64_t seed) {
    for (int attempt = 1; attempt <= kMaxAttemptsPerMove; ++attempt) {
      const VecN q_now = measuredQ();
      // plan on a separate thread, as a planner node would
      auto future = std::async(std::launch::async, [&] {
        if (const Vec3* position = std::get_if<Vec3>(&goal)) return planner_.planToPose(q_now, *position, tool_rotation, seed);
        return planner_.planToConfiguration(q_now, std::get<VecN>(goal), seed);
      });
      const PlannedMotion motion = future.get();
      event("plan to %s (attempt %d): %s in %.1f ms (%s), duration %.2f s, start %s%s", name, attempt, toString(motion.status),
            1e3 * motion.planning_time_s, motion.used_rrt ? "RRT-Connect" : "straight", motion.trajectory.duration(),
            toString(motion.start_validity), motion.start_repaired ? " (repaired)" : "");
      if (!motion.ok()) return false;  // planning is deterministic: retrying the same query cannot help
      stack_.mailbox().post(std::make_shared<JointTrajectoryTask>(motion.trajectory));
      std::this_thread::sleep_for(5ms);
      stack_.supervisor().requestStart();
      if (!waitFor(Mode::Executing, 0.5)) {
        const Mode mode = stack_.supervisor().mode();
        event("start rejected (mode %s)", toString(mode));
        if (mode != Mode::Hold) {  // a fault arrived while planning: the arm may have moved
          ++unplanned_faults_;
          if (!recover()) return false;
        }
        continue;
      }
      waitFor(Mode::Hold, motion.trajectory.duration() + 2.0);
      const Mode ended_in = stack_.supervisor().mode();
      if (ended_in == Mode::Hold && stack_.supervisor().lastReason() == Reason::TaskDone) {
        stack_.mailbox().collect();
        event("reached %s; mailbox retains %zu task(s)", name, stack_.mailbox().retained());
        return true;
      }
      ++unplanned_faults_;
      event("UNPLANNED: move to %s ended in %s (%s); recovering", name, toString(ended_in), toString(stack_.supervisor().lastReason()));
      if (!recover()) {
        event("recovery failed, mode %s", toString(stack_.supervisor().mode()));
        return false;
      }
    }
    return false;
  }

  bool injectStallAndReset() {
    event("injecting a %d ms control-thread stall", kInjectedStallMs);
    runtime_.injectControlStall(kInjectedStallMs);
    const bool tripped = waitFor(Mode::SafeStop, 1.0);
    event("supervisor mode %s (watchdog %s)", toString(stack_.supervisor().mode()), tripped ? "tripped" : "did NOT trip");
    const bool recovered = recover();
    event("operator reset -> mode %s", toString(stack_.supervisor().mode()));
    return tripped && recovered;
  }

  int unplannedFaults() const { return unplanned_faults_; }

 private:
  Runtime& runtime_;
  ControlStack& stack_;
  const MotionPlanner& planner_;
  std::ofstream events_;
  std::chrono::steady_clock::time_point t0_;
  int unplanned_faults_ = 0;
};

void writeSummary(const Options& o, const RuntimeStats& st, const Supervisor& supervisor, int moves_done, int unplanned_faults,
                  bool stall_handled) {
  const auto u64 = [](std::uint64_t v) { return static_cast<unsigned long long>(v); };
  std::printf("\n== real-time summary (%s, %d stress threads)\n", o.tag.c_str(), o.stress_threads);
  std::printf("  SCHED_FIFO granted: control %d, robot %d\n", st.control_fifo, st.robot_fifo);
  std::printf("  control ticks %llu, skipped periods %llu, deadline misses %llu\n", u64(st.control_ticks),
              u64(st.control_skipped_periods), u64(st.control_deadline_misses));
  std::printf("  control wake-up lateness  p50 %.0f us  p99 %.0f us  p99.9 %.0f us  max %.0f us\n", st.control_wake_us.percentile(50),
              st.control_wake_us.percentile(99), st.control_wake_us.percentile(99.9), st.control_wake_us.max());
  std::printf("  control compute           p50 %.0f us  p99 %.0f us  max %.0f us\n", st.control_compute_us.percentile(50),
              st.control_compute_us.percentile(99), st.control_compute_us.max());
  std::printf("  robot wake-up lateness    p99 %.0f us  max %.0f us; steps without a new command %llu of %llu\n",
              st.robot_wake_us.percentile(99), st.robot_wake_us.max(), u64(st.robot_steps_without_new_command), u64(st.robot_steps));
  std::printf("  log records written %llu, dropped %llu\n", u64(st.log_written), u64(st.log_dropped));
  std::printf("  moves completed %d/%d, unplanned faults %d, injected stall handled %d\n", moves_done, kMoves, unplanned_faults,
              stall_handled);
  std::printf("  supervisor transitions:\n");
  for (std::size_t i = 0; i < supervisor.retainedTransitions(); ++i) {
    const Transition& t = supervisor.transition(i);
    std::printf("    t=%7.3f  %-9s -> %-9s (%s)\n", t.time_s, toString(t.from), toString(t.to), toString(t.why));
  }

  std::ofstream f(o.out_dir + "/rt_summary_" + o.tag + ".csv");
  f << "tag,stress_threads,control_fifo,robot_fifo,control_ticks,control_skipped_periods,deadline_misses,wake_p50_us,"
       "wake_p99_us,wake_p999_us,wake_max_us,compute_p50_us,compute_p99_us,compute_max_us,robot_wake_p99_us,"
       "robot_wake_max_us,robot_steps_without_new_command,robot_steps,log_written,log_dropped,moves_completed,"
       "unplanned_faults,stall_handled\n";
  f << o.tag << ',' << o.stress_threads << ',' << st.control_fifo << ',' << st.robot_fifo << ',' << st.control_ticks << ','
    << st.control_skipped_periods << ',' << st.control_deadline_misses << ',' << st.control_wake_us.percentile(50) << ','
    << st.control_wake_us.percentile(99) << ',' << st.control_wake_us.percentile(99.9) << ',' << st.control_wake_us.max() << ','
    << st.control_compute_us.percentile(50) << ',' << st.control_compute_us.percentile(99) << ',' << st.control_compute_us.max()
    << ',' << st.robot_wake_us.percentile(99) << ',' << st.robot_wake_us.max() << ',' << st.robot_steps_without_new_command << ','
    << st.robot_steps << ',' << st.log_written << ',' << st.log_dropped << ',' << moves_done << ',' << unplanned_faults << ','
    << stall_handled << '\n';

  std::ofstream h(o.out_dir + "/rt_hist_" + o.tag + ".csv");
  h << "histogram,bin_start_us,count\n";
  const auto dump = [&](const char* name, const rt::LatencyHistogram& hist) {
    for (std::size_t i = 0; i < hist.counts().size(); ++i)
      if (hist.counts()[i]) h << name << ',' << static_cast<double>(i) * hist.binWidth() << ',' << hist.counts()[i] << '\n';
  };
  dump("control_wake", st.control_wake_us);
  dump("control_compute", st.control_compute_us);
  dump("robot_wake", st.robot_wake_us);

  std::ofstream tr(o.out_dir + "/rt_transitions_" + o.tag + ".csv");
  tr << "time_s,from,to,reason\n";
  for (std::size_t i = 0; i < supervisor.retainedTransitions(); ++i) {
    const Transition& t = supervisor.transition(i);
    tr << t.time_s << ',' << toString(t.from) << ',' << toString(t.to) << ',' << toString(t.why) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse(argc, argv);
  std::filesystem::create_directories(options.out_dir);
  RobotSim sim(G1RTC_MODEL_DIR "/g1_upper_body.xml", SimConfig{}, kSimSeed);
  sim.reset(RobotDescription::homePosture());
  const RobotDescription robot = RobotDescription::fromMujoco(sim.mjModel());
  ControlStack stack(robot, StackConfig::defaultsG1());
  // Built before the threads start: sampling the allowed-collision matrix takes ~0.1 s.
  const MotionPlanner planner(sim.mjModel(), robot);
  RuntimeConfig runtime_config;
  runtime_config.try_realtime = options.try_realtime;
  runtime_config.log_path = options.out_dir + "/rt_log_" + options.tag + ".csv";
  runtime_config.log_every_nth_record = 5;
  Runtime runtime(sim, stack, runtime_config);

  std::atomic<bool> stressing{true};
  std::vector<std::thread> stressors;
  for (int i = 0; i < options.stress_threads; ++i)
    stressors.emplace_back([&] {
      volatile double x = 0;
      while (stressing.load(std::memory_order_relaxed)) x = x + 1e-9;  // pure CPU contention
    });

  Scenario scenario(runtime, stack, planner, options.out_dir + "/rt_events_" + options.tag + ".txt");
  runtime.start();
  stack.supervisor().requestEnable();
  const bool enabled = scenario.waitFor(Mode::Hold, 2.0);
  scenario.event("enabled: %d, mode %s", enabled, toString(stack.supervisor().mode()));

  rbd::DynamicsCache home(robot.model);
  home.update(robot.model, robot.tool, RobotDescription::homePosture(), VecN::Zero(robot.model.nv));
  const Mat3 tool_rotation = home.tool_rotation;
  int moves_done = 0;
  bool stall_handled = false;
  if (enabled) {
    moves_done += scenario.move("under the shelf", Vec3(0.30, -0.28, 0.90), tool_rotation, 10);
    moves_done += scenario.move("above the shelf", Vec3(0.30, -0.28, 1.10), tool_rotation, 11);
    std::this_thread::sleep_for(300ms);
    stall_handled = scenario.injectStallAndReset();
    moves_done += scenario.move("home", RobotDescription::homePosture(), tool_rotation, 12);
  }
  std::this_thread::sleep_for(300ms);
  runtime.stop();
  stressing = false;
  for (auto& t : stressors) t.join();

  writeSummary(options, runtime.stats(), stack.supervisor(), moves_done, scenario.unplannedFaults(), stall_handled);
  return moves_done == kMoves && stall_handled ? 0 : 1;
}
