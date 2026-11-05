#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>

#include "common.hpp"
#include "g1rtc/runtime.hpp"

// ---------------------------------------------------------------- allocation counter
// Counts heap allocations made by the current thread while counting is switched on, to prove
// ControlStack::tick never allocates.
namespace {
thread_local bool g_count = false;
thread_local long g_allocs = 0;
}  // namespace
// Every allocating form is replaced so allocation and deallocation always pair malloc/free (a
// partial replacement mixes with the sanitizer's own operator new and ASan reports a mismatch).
void* operator new(std::size_t n) {
  if (g_count) ++g_allocs;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_count) ++g_allocs;
  return std::malloc(n ? n : 1);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept { return operator new(n, t); }
void* operator new(std::size_t n, std::align_val_t a) {
  if (g_count) ++g_allocs;
  const auto al = static_cast<std::size_t>(a);
  if (void* p = std::aligned_alloc(al, (n + al - 1) / al * al)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t n, std::align_val_t a) { return operator new(n, a); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

using namespace g1test;

TEST(Rt, TripleBufferNeverTearsAndIsMonotonic) {
  struct Payload {
    long a = 0, b = 0;
    double pad[16]{};
  };
  rt::TripleBuffer<Payload> tb;
  std::atomic<bool> done{false}, go{false};
  std::thread prod([&] {
    while (!go.load()) std::this_thread::yield();
    for (long i = 1; i <= 2000000; ++i) {
      Payload& p = tb.writeBuffer();
      p.a = i;
      p.b = -i;
      tb.publish();
    }
    done = true;
  });
  long last = 0, torn = 0, back = 0, reads = 0;
  go = true;
  while (!done.load()) {
    if (tb.update()) {
      const Payload& p = tb.read();
      torn += p.a != -p.b;
      back += p.a < last;
      last = p.a;
      ++reads;
    }
  }
  prod.join();
  tb.update();
  EXPECT_EQ(torn, 0);
  EXPECT_EQ(back, 0);
  EXPECT_EQ(tb.read().a, 2000000);  // the newest value always gets through
  EXPECT_GT(reads, 0);
}

TEST(Rt, SpscQueueKeepsOrderAndCountsDrops) {
  rt::SpscQueue<long, 1024> q;
  std::thread prod([&] {
    for (long i = 0; i < 300000; ++i)
      while (!q.push(i)) std::this_thread::yield();
  });
  long expect = 0, v;
  while (expect < 300000) {
    if (q.pop(v)) {
      EXPECT_EQ(v, expect++);
    }
  }
  prod.join();
  rt::SpscQueue<int, 4> small;
  for (int i = 0; i < 10; ++i) small.push(i);
  EXPECT_EQ(small.dropped(), 6u);
}

TEST(Rt, MailboxNeverFreesATaskInUse) {
  TaskMailbox mb;
  Trajectory tr;
  tr.time_s = {0.0, 0.001};
  tr.q = tr.qd = tr.qdd = {VecN::Zero(10), VecN::Zero(10)};
  auto t1 = std::make_shared<JointTrajectoryTask>(tr);
  std::weak_ptr<JointTrajectoryTask> w1 = t1;
  mb.post(std::move(t1));
  std::uint64_t seq = 0;
  const Task* seen = mb.newest(seq);
  ASSERT_NE(seen, nullptr);
  mb.reportExecuting(seq);  // the control thread is executing task 1
  auto t2 = std::make_shared<JointTrajectoryTask>(tr);
  std::weak_ptr<JointTrajectoryTask> w2 = t2;
  mb.post(std::move(t2));
  mb.newest(seq);  // the control thread has now seen task 2 but still executes task 1
  mb.collect();
  EXPECT_FALSE(w1.expired());
  mb.reportExecuting(seq);  // switched to task 2
  mb.collect();
  EXPECT_TRUE(w1.expired());
  EXPECT_FALSE(w2.expired());
}

TEST(Rt, ControlTickDoesNotAllocate) {
  Rig r(SimConfig{}, home());
  r.lockstep->enable(0.3);
  Figure8Task::Params p;
  r.toolPose(r.sim->trueQ(), p.center, p.rotation);
  p.posture = r.sim->trueQ();
  auto task = std::make_shared<Figure8Task>(p);
  r.stack->mailbox().post(task);
  r.lockstep->run(0.002);
  r.stack->supervisor().requestStart();
  r.lockstep->run(0.5);
  ASSERT_EQ(r.stack->supervisor().mode(), Mode::Executing);
  long allocs = 0;
  for (int k = 0; k < 1000; ++k) {
    r.sim->step();
    r.sim->step();
    const RobotState s = r.sim->state();
    g_allocs = 0;
    g_count = true;
    TorqueCommand cmd = r.stack->tick(s);
    g_count = false;
    allocs += g_allocs;
    r.sim->command(cmd);
  }
  EXPECT_EQ(allocs, 0);
}

TEST(Rt, WatchdogTripsOnControlStallAndResetRecovers) {
#ifdef G1RTC_SLOW_BUILD
  GTEST_SKIP() << "wall-clock 1 kHz test: not meaningful in an unoptimized sanitizer build";
#endif
  RobotSim sim(kScene, SimConfig{}, 1);
  sim.reset(home());
  ControlStack stack(RobotDescription::fromMujoco(sim.mjModel()), StackConfig::defaultsG1());
  RuntimeConfig rc;
  rc.try_realtime = false;
  Runtime rt(sim, stack, rc);
  rt.start();
  auto wait = [&](Mode m, double s) {
    for (int i = 0; i < static_cast<int>(s * 1000) && stack.supervisor().mode() != m; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return stack.supervisor().mode() == m;
  };
  stack.supervisor().requestEnable();
  ASSERT_TRUE(wait(Mode::Hold, 3.0));
  rt.injectControlStall(30);  // the drive watchdog is 5 ms
  EXPECT_TRUE(wait(Mode::SafeStop, 3.0));
  stack.supervisor().requestReset();
  EXPECT_TRUE(wait(Mode::Hold, 3.0));
  rt.stop();
  bool saw_fault = false;
  for (std::size_t i = 0; i < stack.supervisor().retainedTransitions(); ++i)
    saw_fault |= stack.supervisor().transition(i).why == Reason::DriveFault;
  EXPECT_TRUE(saw_fault);
}

TEST(Rt, PeriodicTimerKeepsAverageRate) {
  rt::PeriodicTimer timer(1000000L);
  const double t0 = rt::PeriodicTimer::nowSeconds();
  for (int i = 0; i < 200; ++i) timer.wait();
  const double dt = rt::PeriodicTimer::nowSeconds() - t0;
  EXPECT_NEAR(dt, 0.2, 0.03);  // absolute deadlines: lateness does not accumulate
}
