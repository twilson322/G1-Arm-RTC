// Google Benchmark timings for the pieces of the 1 kHz loop. MuJoCo's mj_forward is listed for
// scale only: it also runs collision detection and the constraint solver, so it is not a like-for-
// like comparison with the controller-side dynamics.
#include <benchmark/benchmark.h>
#include <mujoco/mujoco.h>

#include <random>

#include "g1rtc/lockstep.hpp"
#include "g1rtc/robot_description.hpp"
#include "g1rtc/rt.hpp"

using namespace g1rtc;

namespace {

const char* const kScene = G1RTC_MODEL_DIR "/g1_upper_body.xml";

/// A random but fixed state, the same for every benchmark.
struct Fixture {
  RobotSim sim{kScene, SimConfig{}, 1};
  RobotDescription robot = RobotDescription::fromMujoco(sim.mjModel());
  VecN q, qd, qdd;
  Fixture() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> unit(-0.5, 0.5);
    const int n = robot.model.nv;
    q = qd = qdd = VecN::Zero(n);
    for (int i = 0; i < n; ++i) {
      q[i] = unit(rng);
      qd[i] = 2 * unit(rng);
      qdd[i] = 4 * unit(rng);
    }
  }
};

Fixture& fixture() {
  static Fixture f;
  return f;
}

/// One full control tick on a fresh simulator state each iteration. The plant step is excluded
/// with PauseTiming, whose own overhead (well under 1 us) is small next to a tick. With
/// `flush_to_zero` the tick runs with FTZ/DAZ set, as the runtime's control thread does; the
/// simulator steps without it either way.
void benchmarkTick(benchmark::State& st, bool with_figure8, bool flush_to_zero) {
  RobotSim sim(kScene, SimConfig{}, 2);
  sim.reset(RobotDescription::homePosture());
  const RobotDescription robot = RobotDescription::fromMujoco(sim.mjModel());
  ControlStack stack(robot, StackConfig::defaultsG1());
  Lockstep lockstep(sim, stack);
  lockstep.enable(0.5);
  if (with_figure8) {
    Figure8Task::Params p;
    rbd::DynamicsCache dyn(robot.model);
    dyn.update(robot.model, robot.tool, sim.state().q, sim.state().qd);
    p.center = dyn.tool_position;
    p.rotation = dyn.tool_rotation;
    p.posture = sim.state().q;
    p.laps = 100000;  // never finishes during the benchmark
    stack.mailbox().post(std::make_shared<Figure8Task>(p));
    lockstep.run(2 * Lockstep::kControlPeriodS);
    stack.supervisor().requestStart();
    lockstep.run(2 * Lockstep::kControlPeriodS);
    if (stack.supervisor().mode() != Mode::Executing) st.SkipWithError("figure-8 did not start");
  }
  const int physics_steps_per_tick = static_cast<int>(std::lround(Lockstep::kControlPeriodS / sim.physicsDt()));
  const unsigned saved_float_control = rt::floatControlWord();
  for (auto _ : st) {
    st.PauseTiming();
    for (int i = 0; i < physics_steps_per_tick; ++i) sim.step();
    if (flush_to_zero) rt::enableFlushToZero();
    st.ResumeTiming();
    TorqueCommand command = stack.tick(sim.state());
    benchmark::DoNotOptimize(command.tau.data());
    st.PauseTiming();
    rt::setFloatControlWord(saved_float_control);
    sim.command(command);
    st.ResumeTiming();
  }
}

}  // namespace

static void BM_ForwardKinematics(benchmark::State& st) {
  const Fixture& f = fixture();
  rbd::Kinematics<double> kin(f.robot.model);
  for (auto _ : st) {
    rbd::forwardKinematics(f.robot.model, f.q, kin);
    benchmark::DoNotOptimize(kin.origin.back());
  }
}
BENCHMARK(BM_ForwardKinematics);

static void BM_Rnea(benchmark::State& st) {
  const Fixture& f = fixture();
  rbd::Kinematics<double> kin(f.robot.model);
  rbd::Workspace<double> work(f.robot.model);
  rbd::forwardKinematics(f.robot.model, f.q, kin);
  VecN tau(f.robot.model.nv);
  for (auto _ : st) {
    rbd::rnea(f.robot.model, kin, f.qd, f.qdd, true, work, tau);
    benchmark::DoNotOptimize(tau.data());
  }
}
BENCHMARK(BM_Rnea);

static void BM_Crba(benchmark::State& st) {
  const Fixture& f = fixture();
  rbd::Kinematics<double> kin(f.robot.model);
  rbd::Workspace<double> work(f.robot.model);
  rbd::forwardKinematics(f.robot.model, f.q, kin);
  MatN M(f.robot.model.nv, f.robot.model.nv);
  for (auto _ : st) {
    rbd::crba(f.robot.model, kin, work, M);
    benchmark::DoNotOptimize(M.data());
  }
}
BENCHMARK(BM_Crba);

static void BM_ObserverBiasWithDualNumbers(benchmark::State& st) {
  const Fixture& f = fixture();
  rbd::Kinematics<double> kin(f.robot.model);
  rbd::Workspace<double> work(f.robot.model);
  rbd::Kinematics<Dual> kin_dual(f.robot.model);
  rbd::Workspace<Dual> work_dual(f.robot.model);
  for (auto _ : st) benchmark::DoNotOptimize(rbd::observerBias(f.robot.model, f.q, f.qd, kin, work, kin_dual, work_dual));
}
BENCHMARK(BM_ObserverBiasWithDualNumbers);

static void BM_DynamicsCacheUpdate(benchmark::State& st) {
  const Fixture& f = fixture();
  rbd::DynamicsCache dyn(f.robot.model);
  for (auto _ : st) {
    dyn.update(f.robot.model, f.robot.tool, f.q, f.qd);
    benchmark::DoNotOptimize(dyn.J.data());
  }
}
BENCHMARK(BM_DynamicsCacheUpdate);

// Hold without FTZ is the slow case: with the arm at rest, quantities that decay toward zero
// reach the denormal range, where every operation on them is several times slower.
static void BM_ControlTickHold(benchmark::State& st) { benchmarkTick(st, false, false); }
BENCHMARK(BM_ControlTickHold);
static void BM_ControlTickHoldFlushToZero(benchmark::State& st) { benchmarkTick(st, false, true); }
BENCHMARK(BM_ControlTickHoldFlushToZero);
static void BM_ControlTickOperationalSpace(benchmark::State& st) { benchmarkTick(st, true, false); }
BENCHMARK(BM_ControlTickOperationalSpace);
static void BM_ControlTickOperationalSpaceFlushToZero(benchmark::State& st) { benchmarkTick(st, true, true); }
BENCHMARK(BM_ControlTickOperationalSpaceFlushToZero);

static void BM_MuJoCoForwardForScale(benchmark::State& st) {
  const Fixture& f = fixture();
  mjData* d = mj_makeData(f.sim.mjModel());
  for (int i = 0; i < f.robot.model.nv; ++i) {
    d->qpos[i] = f.q[i];
    d->qvel[i] = f.qd[i];
  }
  for (auto _ : st) mj_forward(f.sim.mjModel(), d);
  mj_deleteData(d);
}
BENCHMARK(BM_MuJoCoForwardForScale);

BENCHMARK_MAIN();
