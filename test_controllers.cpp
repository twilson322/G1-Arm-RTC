#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <random>

#include "common.hpp"

using namespace g1test;

TEST(Controllers, GravityCompensatedHoldStaysPut) {
  Rig r(idealSim(), home());
  r.lockstep->enable(2.0);
  EXPECT_EQ(r.stack->supervisor().mode(), Mode::Hold);
  // the hold pose is captured when Hold is entered, after a few ms of Enabling
  EXPECT_LT((r.sim->trueQ() - r.stack->supervisor().holdPose()).cwiseAbs().maxCoeff(), 1e-3);
  EXPECT_LT(r.sim->trueQd().norm(), 1e-3);
}

TEST(Controllers, ComputedTorqueTracksJointSinusoidWithExactModel) {
  Rig r(idealSim(), home());
  r.lockstep->enable(0.3);
  Trajectory tr;
  const VecN q0 = r.sim->trueQ();
  for (int k = 0; k <= 3000; ++k) {
    const double t = 0.001 * k, w = 2 * std::numbers::pi * 0.5;
    // smooth start: a(t) = 0.3 (1 - cos(w t)) on two arm joints
    VecN q = q0, qd = VecN::Zero(10), qdd = VecN::Zero(10);
    for (int j : {3, 6}) {
      q[j] += 0.3 * (1 - std::cos(w * t));
      qd[j] = 0.3 * w * std::sin(w * t);
      qdd[j] = 0.3 * w * w * std::cos(w * t);
    }
    tr.time_s.push_back(t);
    tr.q.push_back(q);
    tr.qd.push_back(qd);
    tr.qdd.push_back(qdd);
  }
  auto task = std::make_shared<JointTrajectoryTask>(tr, TrackingMode::ComputedTorque);
  double max_err = 0;
  VecN qr, qdr, qddr;
  ASSERT_TRUE(r.lockstep->execute(task, 5.0, [&](const RobotState&, const TorqueCommand&) {
    if (r.stack->supervisor().mode() != Mode::Executing) return;
    tr.sample(r.stack->taskTime(), qr, qdr, qddr);
    max_err = std::max(max_err, (r.sim->trueQ() - qr).cwiseAbs().maxCoeff());
  }));
  // exact model, ideal sensing: what is left is the 1.5 ms command latency
  EXPECT_LT(max_err, 3e-3);
}

TEST(Controllers, OscNullSpaceIsDynamicallyConsistent) {
  Rig r(idealSim(), home());
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-0.3, 0.3);
  for (int trial = 0; trial < 10; ++trial) {
    VecN q = home();
    for (int i = 0; i < 10; ++i) q[i] += u(rng);
    rbd::DynamicsCache dc(r.robot.model);
    dc.update(r.robot.model, r.robot.tool, q, VecN::Zero(10));
    Reference ref = Reference::hold(VecN(q + VecN::Constant(10, 0.1)));
    const Vec3 f(0, 0, 0);
    const VecN qd = VecN::Zero(10);
    ControlContext c{r.robot.model, dc, q, qd, f, 0.001};
    // exact Lambda: a null-space torque produces no task acceleration, J M^-1 N^T tau0 = 0
    TaskGains exact;
    exact.lambda_regularization = 0.0;
    OscTerms t;
    t.compute(c, ref, exact);
    VecN acc = dc.M_cholesky.solve(t.tau_posture_null_space);
    EXPECT_LT((dc.J * acc).norm(), 1e-8 * std::max(1.0, t.tau_posture_null_space.norm()));
    // the default regularization (1e-3) leaks a little posture torque into the task
    t.compute(c, ref, TaskGains{});
    acc = dc.M_cholesky.solve(t.tau_posture_null_space);
    EXPECT_LT((dc.J * acc).norm(), 0.05 * (dc.J * dc.M_cholesky.solve(VecN(dc.M * (ref.q - q)))).norm() + 1e-6);
  }
}

TEST(Controllers, OscReachesCartesianSetpoint) {
  Rig r(idealSim(), home());
  r.lockstep->enable(0.3);
  Figure8Task::Params p;
  r.toolPose(r.sim->trueQ(), p.center, p.rotation);
  p.posture = r.sim->trueQ();
  p.first_amplitude_m = 0.05;
  p.second_amplitude_m = 0.03;
  p.laps = 1;
  auto task = std::make_shared<Figure8Task>(p);
  double max_err = 0;
  Reference ref;
  ref.q = p.posture;
  ASSERT_TRUE(r.lockstep->execute(task, 10.0, [&](const RobotState&, const TorqueCommand&) {
    if (r.stack->supervisor().mode() != Mode::Executing) return;
    task->reference(r.stack->taskTime(), ref);
    max_err = std::max(max_err, (r.stack->dynamics().tool_position - ref.tool_position).norm());
  }));
  EXPECT_LT(max_err, 2e-3);
  Vec3 x;
  Mat3 R;
  r.toolPose(r.sim->trueQ(), x, R);
  EXPECT_LT((x - p.center).norm(), 1e-3);  // the figure ends where it started
}

TEST(Controllers, HybridForceRegulatesContactForce) {
  SimConfig sc;  // realistic sensing and friction: this is the case the PI loop is for
  Rig probe(sc, home());
  Vec3 xh;
  Mat3 Rh;
  probe.toolPose(home(), xh, Rh);
  const Vec3 surface(0.40, -0.22, 0.65);
  const IkResult ik = solveIk(probe.robot.model, probe.robot.tool, surface + Vec3(0, 0, 0.03), Rh, home(), home());
  ASSERT_TRUE(ik.converged);
  Rig r(sc, ik.q);
  r.lockstep->enable(0.3);
  WipeTask::Params p;
  p.start_point = surface;
  p.rotation = Rh;
  p.posture = ik.q;
  p.radius_m = 0.0;  // press in place
  p.laps = 1;
  p.push_force_n = 8.0;
  auto task = std::make_shared<WipeTask>(p);
  double sum = 0;
  int n = 0;
  ASSERT_TRUE(r.lockstep->execute(task, 10.0, [&](const RobotState&, const TorqueCommand&) {
    if (r.stack->supervisor().mode() != Mode::Executing) return;
    if (r.stack->taskTime() > 2.0) {
      sum += r.sim->toolContactForce().z();
      ++n;
    }
  }));
  ASSERT_GT(n, 500) << r.transitions();
  EXPECT_NEAR(sum / n, 8.0, 0.4);
}

TEST(Controllers, SaturateClampsAndCounts) {
  Rig r(idealSim(), home());
  VecN tau = VecN::Zero(10);
  tau[0] = 1000;
  tau[9] = -1000;
  tau[4] = 3.0;
  EXPECT_EQ(saturate(r.robot.model, tau), 2);
  EXPECT_DOUBLE_EQ(tau[0], r.robot.model.tau_max[0]);
  EXPECT_DOUBLE_EQ(tau[9], -r.robot.model.tau_max[9]);
  EXPECT_DOUBLE_EQ(tau[4], 3.0);
}
