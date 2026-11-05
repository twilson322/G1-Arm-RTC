#include <gtest/gtest.h>

#include <random>

#include "g1rtc/lockstep.hpp"

#include "common.hpp"

using namespace g1test;

TEST(Observer, ResidualIsZeroInFreeMotionWithExactModel) {
  StackConfig cfg = StackConfig::defaultsG1();
  cfg.observer.model_joint_friction = false;
  cfg.observer.drive_velocity_filter_hz = 0.0;  // ideal sensing: nothing to match
  cfg.detect_collisions = false;
  Rig r(idealSim(), home(), cfg);
  r.lockstep->enable(0.3);
  Figure8Task::Params p;
  r.toolPose(r.sim->trueQ(), p.center, p.rotation);
  p.posture = r.sim->trueQ();
  p.laps = 1;
  auto task = std::make_shared<Figure8Task>(p);
  double mx = 0;
  r.lockstep->execute(task, 10, [&](const RobotState&, const TorqueCommand&) {
    mx = std::max(mx, r.stack->residual().cwiseAbs().maxCoeff());
  });
  EXPECT_LT(mx, 0.02);  // N m, against thresholds of 0.16-0.91 N m
}

TEST(Observer, ResidualStaysSmallWhenTheControlPeriodJitters) {
  // In the threaded runtime the control thread sometimes wakes late and sees two robot cycles at
  // once. The observer integrates with the measured period and the drives' torque impulse, so an
  // irregular period must not show up as external torque. With a fixed nominal 1 ms it did, enough
  // to trip the collision detector.
  StackConfig cfg = StackConfig::defaultsG1();
  cfg.observer.model_joint_friction = false;
  cfg.observer.drive_velocity_filter_hz = 0.0;
  cfg.detect_collisions = false;
  Rig r(idealSim(), home(), cfg);
  r.lockstep->enable(0.3);
  Figure8Task::Params p;
  r.toolPose(r.sim->trueQ(), p.center, p.rotation);
  p.posture = r.sim->trueQ();
  p.laps = 1;
  r.stack->mailbox().post(std::make_shared<Figure8Task>(p));
  r.lockstep->run(0.002);
  r.stack->supervisor().requestStart();
  r.lockstep->run(0.002);
  ASSERT_EQ(r.stack->supervisor().mode(), Mode::Executing);
  std::mt19937 rng(4);
  std::uniform_int_distribution<int> physics_steps(2, 4);  // 1, 1.5 or 2 ms between ticks
  double max_residual = 0;
  std::uint64_t seq = 1000;
  for (int k = 0; k < 10000 && r.stack->supervisor().mode() == Mode::Executing; ++k) {
    for (int i = physics_steps(rng); i > 0; --i) r.sim->step();
    TorqueCommand command = r.stack->tick(r.sim->state());
    command.seq = ++seq;
    r.sim->command(command);
    max_residual = std::max(max_residual, r.stack->residual().cwiseAbs().maxCoeff());
  }
  EXPECT_EQ(r.stack->supervisor().lastReason(), Reason::TaskDone) << r.transitions();
  EXPECT_LT(max_residual, 0.05);  // N m; the smallest collision threshold is 0.16
}

TEST(Observer, ResidualConvergesToExternalJointTorque) {
  StackConfig cfg = StackConfig::defaultsG1();
  cfg.observer.model_joint_friction = false;
  cfg.observer.drive_velocity_filter_hz = 0.0;
  cfg.detect_collisions = false;
  Rig r(idealSim(), home(), cfg);
  r.lockstep->enable(0.3);
  const int tool_body = r.sim->bodyId("tool");
  r.sim->scheduleDisturbance({tool_body, Vec3(5.0, -3.0, 4.0), r.sim->time() + 0.01, 10.0});
  r.lockstep->run(0.3);  // ~9 observer time constants
  const VecN truth = r.sim->trueExternalTorque();
  ASSERT_GT(truth.norm(), 0.5);
  EXPECT_LT((r.stack->residual() - truth).norm() / truth.norm(), 0.03);
}

TEST(Observer, DetectorNeedsPersistence) {
  CollisionDetector::Params p;
  p.threshold_nm = VecN::Constant(3, 1.0);
  p.persistence_ticks = 3;
  CollisionDetector d(p);
  VecN r = VecN::Zero(3);
  r[1] = 2.0;
  EXPECT_FALSE(d.update(r));
  EXPECT_FALSE(d.update(r));
  r[1] = 0.0;
  EXPECT_FALSE(d.update(r));  // the run of samples is broken
  r[1] = 2.0;
  r[2] = 1.5;
  EXPECT_FALSE(d.update(r));
  EXPECT_FALSE(d.update(r));
  EXPECT_TRUE(d.update(r));
  EXPECT_FALSE(d.update(r));  // fires once until re-armed
  d.rearm();
  EXPECT_FALSE(d.update(r));
  EXPECT_FALSE(d.update(r));
  EXPECT_TRUE(d.update(r));
}

TEST(Observer, ResidualFittingIsolatesTheStruckLink) {
  Rig r(idealSim(), home());
  const std::vector<std::string> names = {"torso_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
                                          "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
                                          "right_wrist_pitch_link", "right_wrist_yaw_link", "tool"};
  std::vector<int> ids;
  for (const auto& n : names) ids.push_back(r.robot.model.bodyId(n));
  std::mt19937 rng(5);
  std::normal_distribution<double> g(0.0, 1.0);
  rbd::Kinematics<double> k(r.robot.model);
  int correct = 0, near = 0, total = 0;
  for (int trial = 0; trial < 20; ++trial) {
    VecN q = home();
    for (int i = 0; i < 10; ++i) q[i] += 0.2 * g(rng);
    rbd::forwardKinematics(r.robot.model, q, k);
    for (std::size_t c = 1; c < ids.size(); ++c) {  // skip the torso: only waist joints see it
      Mat6N J;
      rbd::pointJacobian(r.robot.model, k, ids[c], k.com[static_cast<std::size_t>(ids[c])], J);
      const Vec3 f(10 * g(rng), 10 * g(rng), 10 * g(rng));
      const VecN tau = J.topRows<3>().transpose() * f;
      const int got = isolateContactBody(r.robot.model, k, tau, VecN::Ones(10), ids);
      correct += got == ids[c];
      for (int d : {-1, 0, 1}) {
        const auto j = static_cast<std::size_t>(static_cast<int>(c) + d);
        if (j < ids.size() && got == ids[j]) ++near;
      }
      ++total;
    }
  }
  // Noise-free residuals, and still not 100 %: the shoulder-pitch link's COM sits almost on its
  // own joint axis, so a push there produces almost no torque on that joint and looks like a push
  // on the torso (16 of 20 cases); the wrist links have nearly coincident COMs. Exact isolation
  // was 134/160 and one case was off by two links when this test was written.
  EXPECT_GE(correct, static_cast<int>(0.8 * total)) << correct << "/" << total;
  EXPECT_GE(near, total - 2) << near << "/" << total;
}

TEST(Observer, PayloadRlsRecoversMassAndCom) {
  Rig r(idealSim(), home());
  PayloadEstimator::Params params;
  params.forgetting = 0.99999;
  params.estimate_com = true;
  PayloadEstimator est(params);
  const Eigen::Vector4d theta(0.8, 0.8 * 0.04, 0.8 * -0.01, 0.8 * 0.02);
  std::mt19937 rng(9);
  std::normal_distribution<double> g(0.0, 1.0);
  const rbd::Model& model = r.robot.model;
  rbd::DynamicsCache dc(model);
  for (int k = 0; k < 400; ++k) {
    VecN q = home();
    for (int i = 0; i < 10; ++i) q[i] += 0.4 * g(rng);  // orientation diversity makes c identifiable
    dc.update(model, r.robot.tool, q, VecN::Zero(10));
    const VecN residual = PayloadEstimator::regressor(dc, model.gravity) * theta;
    est.update(dc, residual, model.gravity);
  }
  EXPECT_NEAR(est.massKg(), 0.8, 1e-3);
  EXPECT_LT((est.comToolFrame() - Vec3(0.04, -0.01, 0.02)).norm(), 1e-3);
}

TEST(Observer, ToolWrenchCompensationReadsAppliedForce) {
  SimConfig sc = idealSim();
  StackConfig cfg = StackConfig::defaultsG1();
  cfg.detect_collisions = false;  // the applied force is not a collision here
  Rig r(sc, home(), cfg);
  r.lockstep->enable(0.3);
  EXPECT_LT(r.stack->toolContactForce().norm(), 0.05);  // tared, tool weight removed
  const int tool_body = r.sim->bodyId("tool");
  const Vec3 F(2.0, -1.0, 3.0);
  r.sim->scheduleDisturbance({tool_body, F, r.sim->time() + 0.001, 10.0});
  r.lockstep->run(0.5);
  // a static robot under hold: the wrist sensor sees the applied force (plus small inertial terms)
  EXPECT_LT((r.stack->toolContactForce() - F).norm(), 0.15);
}
