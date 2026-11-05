// The hand-written dynamics against MuJoCo, at random states of the 10-DOF G1 upper body.
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>

#include <random>

#include "g1rtc/rbd.hpp"

using namespace g1rtc;

namespace {

struct Mj : ::testing::Test {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  rbd::Model model;
  std::mt19937 rng{42};

  void SetUp() override {
    char err[512] = "";
    m = mj_loadXML(G1RTC_MODEL_DIR "/g1_upper_body.xml", nullptr, err, sizeof(err));
    ASSERT_NE(m, nullptr) << err;
    d = mj_makeData(m);
    model = rbd::Model::fromMujoco(m);
  }
  void TearDown() override {
    mj_deleteData(d);
    mj_deleteModel(m);
  }
  void randomState(VecN& q, VecN& qd, VecN& qdd) {
    std::uniform_real_distribution<double> u(0.0, 1.0), v(-2.0, 2.0);
    q.resize(model.nv);
    qd.resize(model.nv);
    qdd.resize(model.nv);
    for (int i = 0; i < model.nv; ++i) {
      q[i] = model.q_min[i] + u(rng) * (model.q_max[i] - model.q_min[i]);
      qd[i] = v(rng);
      qdd[i] = 3.0 * v(rng);
    }
    for (int i = 0; i < model.nv; ++i) {
      d->qpos[i] = q[i];
      d->qvel[i] = qd[i];
      d->qacc[i] = qdd[i];
    }
    mj_forward(m, d);
  }
};

TEST_F(Mj, ForwardKinematicsMatchesMujoco) {
  rbd::Kinematics<double> k(model);
  VecN q, qd, qdd;
  for (int trial = 0; trial < 20; ++trial) {
    randomState(q, qd, qdd);
    rbd::forwardKinematics(model, q, k);
    for (int b = 1; b < m->nbody; ++b) {
      const auto ub = static_cast<std::size_t>(b);
      const Vec3 p(d->xpos[3 * b], d->xpos[3 * b + 1], d->xpos[3 * b + 2]);
      const Mat3 R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(d->xmat + 9 * b);
      EXPECT_LT((k.origin[ub] - p).norm(), 1e-12) << model.bodies[ub].name;
      EXPECT_LT((k.rotation[ub] - R).norm(), 1e-12) << model.bodies[ub].name;
    }
  }
}

TEST_F(Mj, MassMatrixMatchesMujoco) {
  rbd::Kinematics<double> k(model);
  rbd::Workspace<double> w(model);
  MatN M;
  std::vector<double> Mmj(static_cast<std::size_t>(m->nv * m->nv));
  VecN q, qd, qdd;
  for (int trial = 0; trial < 20; ++trial) {
    randomState(q, qd, qdd);
    rbd::forwardKinematics(model, q, k);
    rbd::crba(model, k, w, M);
    mj_fullM(m, d, Mmj.data());
    const Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> Mref(Mmj.data(), m->nv, m->nv);
    EXPECT_LT((M - Mref).cwiseAbs().maxCoeff(), 1e-10);
  }
}

TEST_F(Mj, BiasAndInverseDynamicsMatchMujoco) {
  rbd::Kinematics<double> k(model);
  rbd::Workspace<double> w(model);
  MatN M;
  VecN h, tau;
  VecN q, qd, qdd;
  for (int trial = 0; trial < 20; ++trial) {
    randomState(q, qd, qdd);
    rbd::forwardKinematics(model, q, k);
    rbd::rnea(model, k, qd, VecN::Zero(model.nv), true, w, h);
    const Eigen::Map<const Eigen::VectorXd> bias(d->qfrc_bias, m->nv);
    EXPECT_LT((h - bias).cwiseAbs().maxCoeff(), 1e-10);
    // full inverse dynamics against M qdd + bias, with MuJoCo's M (armature included)
    rbd::rnea(model, k, qd, qdd, true, w, tau);
    std::vector<double> Mq(static_cast<std::size_t>(m->nv));
    mj_mulM(m, d, Mq.data(), qdd.data());  // mj_forward overwrote d->qacc
    for (int i = 0; i < m->nv; ++i) EXPECT_NEAR(tau[i], Mq[static_cast<std::size_t>(i)] + bias[i], 1e-9);
    // RNEA and CRBA agree with each other
    rbd::crba(model, k, w, M);
    EXPECT_LT((tau - (M * qdd + h)).cwiseAbs().maxCoeff(), 1e-10);
  }
}

TEST_F(Mj, JacobianAndJdotMatchMujoco) {
  rbd::Kinematics<double> k(model);
  rbd::Workspace<double> w(model);
  const rbd::Frame tool = model.site(m, "tool_tip");
  Mat6N J;
  VecN q, qd, qdd;
  std::vector<double> jp(static_cast<std::size_t>(3 * m->nv)), jr(jp.size()), jdp(jp.size()), jdr(jp.size());
  for (int trial = 0; trial < 20; ++trial) {
    randomState(q, qd, qdd);
    rbd::forwardKinematics(model, q, k);
    const auto ub = static_cast<std::size_t>(tool.body);
    const Vec3 x = k.origin[ub] + k.rotation[ub] * tool.offset_local;
    rbd::pointJacobian(model, k, tool.body, x, J);
    mj_jac(m, d, jp.data(), jr.data(), x.data(), tool.body);
    mj_jacDot(m, d, jdp.data(), jdr.data(), x.data(), tool.body);
    using RM = Eigen::Matrix<double, 3, -1, Eigen::RowMajor>;
    const Eigen::Map<const RM> Jp(jp.data(), 3, m->nv), Jr(jr.data(), 3, m->nv);
    const Eigen::Map<const RM> Jdp(jdp.data(), 3, m->nv), Jdr(jdr.data(), 3, m->nv);
    EXPECT_LT((J.topRows<3>() - Jp).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((J.bottomRows<3>() - Jr).cwiseAbs().maxCoeff(), 1e-12);
    const Vec6 jdqd = rbd::pointJdotQd(model, k, tool.body, x, qd, w);
    EXPECT_LT((jdqd.head<3>() - Jdp * qd).norm(), 1e-9);
    EXPECT_LT((jdqd.tail<3>() - Jdr * qd).norm(), 1e-9);
  }
}

TEST_F(Mj, ObserverBiasUsesExactMdot) {
  rbd::Kinematics<double> k(model), k2(model);
  rbd::Workspace<double> w(model);
  rbd::Kinematics<Dual> kd(model);
  rbd::Workspace<Dual> wd(model);
  VecN q, qd, qdd;
  for (int trial = 0; trial < 10; ++trial) {
    randomState(q, qd, qdd);
    const VecN beta = rbd::observerBias(model, q, qd, k, w, kd, wd);
    // central finite difference of M along qd
    const double h = 1e-6;
    MatN Mp, Mm;
    rbd::forwardKinematics(model, VecN(q + h * qd), k2);
    rbd::crba(model, k2, w, Mp);
    rbd::forwardKinematics(model, VecN(q - h * qd), k2);
    rbd::crba(model, k2, w, Mm);
    const MatN Mdot = (Mp - Mm) / (2 * h);
    VecN Cqd, g;
    rbd::forwardKinematics(model, q, k);
    rbd::rnea(model, k, qd, VecN::Zero(model.nv), false, w, Cqd);
    rbd::rnea(model, k, VecN::Zero(model.nv), VecN::Zero(model.nv), true, w, g);
    const VecN ref = g - (Mdot * qd - Cqd);
    EXPECT_LT((beta - ref).cwiseAbs().maxCoeff(), 1e-6);
    // passivity: qd^T (Mdot - 2C) qd = 0  <=>  qd.(Mdot qd) = 2 qd.(C qd)
    EXPECT_NEAR(qd.dot(Mdot * qd), 2.0 * qd.dot(Cqd), 1e-5);
  }
}

TEST_F(Mj, PayloadMergeMatchesMujocoGravity) {
  const int tool = model.bodyId("tool");
  const Vec3 cp(0.02, -0.01, 0.03);
  const double mp = 0.8;
  rbd::Model withPayload = model;
  withPayload.addPayload(tool, mp, cp);
  // same payload in MuJoCo: body mass and COM change, gravity only depends on those
  const double m0 = m->body_mass[tool];
  const Vec3 c0(m->body_ipos[3 * tool], m->body_ipos[3 * tool + 1], m->body_ipos[3 * tool + 2]);
  const Vec3 c = (m0 * c0 + mp * cp) / (m0 + mp);
  m->body_mass[tool] = m0 + mp;
  for (int i = 0; i < 3; ++i) m->body_ipos[3 * tool + i] = c[i];
  m->body_sameframe[tool] = 0;  // the compiler flagged the inertial frame as coincident; undo that
  rbd::Kinematics<double> k(withPayload);
  rbd::Workspace<double> w(withPayload);
  VecN q, qd, qdd, g;
  for (int trial = 0; trial < 10; ++trial) {
    randomState(q, qd, qdd);
    for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
    mj_forward(m, d);
    rbd::forwardKinematics(withPayload, q, k);
    rbd::rnea(withPayload, k, VecN::Zero(model.nv), VecN::Zero(model.nv), true, w, g);
    for (int i = 0; i < m->nv; ++i) EXPECT_NEAR(g[i], d->qfrc_bias[i], 1e-10);
  }
  EXPECT_NEAR(withPayload.totalMass(), model.totalMass() + mp, 1e-12);
}

}  // namespace
