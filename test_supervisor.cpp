#include <gtest/gtest.h>

#include <thread>

#include "common.hpp"

using namespace g1test;

namespace {
struct Fixture : ::testing::Test {
  rbd::Model model = rbd::Model::fromXml(G1RTC_MODEL_DIR "/g1_upper_body.xml");
  VecN q = home(), qd = VecN::Zero(10);
  SupervisorConfig cfg;
  Mode step(Supervisor& s, double t, bool fault = false, bool coll = false, bool done = false, bool start_ok = true) {
    return s.step({t, q, qd, fault, coll, done, start_ok});
  }
  void toHold(Supervisor& s, double& t) {
    s.requestEnable();
    step(s, t += 0.001, true);  // drive still disabled while the reset is in flight
    for (int i = 0; i < 60; ++i) step(s, t += 0.001);
    ASSERT_EQ(s.mode(), Mode::Hold);
  }
};
}  // namespace

TEST_F(Fixture, EnableSequence) {
  Supervisor s(model, cfg);
  double t = 0;
  EXPECT_EQ(step(s, t), Mode::Disabled);
  s.requestEnable();
  EXPECT_EQ(step(s, t += 0.001), Mode::Enabling);
  EXPECT_TRUE(s.wantsDriveReset());
  for (int i = 0; i < 40; ++i) step(s, t += 0.001);
  EXPECT_EQ(s.mode(), Mode::Enabling);  // enable_time is 50 ms
  for (int i = 0; i < 20; ++i) step(s, t += 0.001);
  EXPECT_EQ(s.mode(), Mode::Hold);
}

TEST_F(Fixture, StartNeedsTaskAtCurrentPose) {
  Supervisor s(model, cfg);
  double t = 0;
  toHold(s, t);
  s.requestStart();
  EXPECT_EQ(step(s, t += 0.001, false, false, false, false), Mode::Hold);
  s.requestStart();
  EXPECT_EQ(step(s, t += 0.001), Mode::Executing);
  EXPECT_EQ(step(s, t += 0.001, false, false, true), Mode::Hold);
}

TEST_F(Fixture, CollisionReflexThenHoldAtNewPose) {
  cfg.reflex_duration_s = 0.1;
  Supervisor s(model, cfg);
  double t = 0;
  toHold(s, t);
  s.requestStart();
  step(s, t += 0.001);
  q[3] += 0.02;
  EXPECT_EQ(step(s, t += 0.001, false, true), Mode::Reflex);
  q[3] += 0.03;  // the arm moves away during the reflex
  for (int i = 0; i < 99; ++i) step(s, t += 0.001);
  EXPECT_EQ(s.mode(), Mode::Reflex);
  step(s, t += 0.001);
  EXPECT_EQ(s.mode(), Mode::Hold);
  EXPECT_DOUBLE_EQ(s.holdPose()[3], q[3]);  // bumpless: holds where the reflex left it
}

TEST_F(Fixture, IgnoreReflexKeepsExecuting) {
  cfg.reflex = ReflexKind::Ignore;
  Supervisor s(model, cfg);
  double t = 0;
  toHold(s, t);
  s.requestStart();
  step(s, t += 0.001);
  EXPECT_EQ(step(s, t += 0.001, false, true), Mode::Executing);
}

TEST_F(Fixture, LimitsAndFaultsForceSafeStop) {
  Supervisor s(model, cfg);
  double t = 0;
  toHold(s, t);
  q[6] = model.q_max[6] - 0.001;  // inside the 10 mrad margin
  qd[6] = 0.5;                    // and moving toward the limit
  EXPECT_EQ(step(s, t += 0.001), Mode::SafeStop);
  EXPECT_EQ(s.lastReason(), Reason::JointLimit);
  q = home();
  qd.setZero();
  step(s, t += 0.001);
  EXPECT_EQ(s.mode(), Mode::SafeStop);  // latched until reset
  s.requestReset();
  EXPECT_EQ(step(s, t += 0.001), Mode::Enabling);
  for (int i = 0; i < 60; ++i) step(s, t += 0.001);
  ASSERT_EQ(s.mode(), Mode::Hold);
  qd[4] = 1.1 * model.qd_max[4];
  EXPECT_EQ(step(s, t += 0.001), Mode::SafeStop);
  qd.setZero();
  s.requestReset();
  step(s, t += 0.001);
  for (int i = 0; i < 60; ++i) step(s, t += 0.001);
  ASSERT_EQ(s.mode(), Mode::Hold);
  EXPECT_EQ(step(s, t += 0.001, true), Mode::SafeStop);
  EXPECT_EQ(s.transition(s.retainedTransitions() - 1).why, Reason::DriveFault);
  EXPECT_EQ(s.lastReason(), Reason::DriveFault);
}

TEST_F(Fixture, RestingInsideTheLimitMarginIsAllowed) {
  // After a drive fault the arm can sag onto a joint stop. An absolute check then tripped again on
  // the first Hold tick and the robot could never be re-enabled.
  Supervisor s(model, cfg);
  double t = 0;
  q[6] = model.q_max[6] - 0.001;
  toHold(s, t);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(step(s, t += 0.001), Mode::Hold);
  qd[6] = -0.5;  // moving away from the limit is fine too
  EXPECT_EQ(step(s, t += 0.001), Mode::Hold);
  qd[6] = 0.5;
  EXPECT_EQ(step(s, t += 0.001), Mode::SafeStop);
}

TEST_F(Fixture, RequestsFromAnotherThreadAreSeen) {
  Supervisor s(model, cfg);
  std::thread op([&] { s.requestEnable(); });
  op.join();
  EXPECT_EQ(step(s, 0.001), Mode::Enabling);
}
