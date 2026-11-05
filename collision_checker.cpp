#include <mujoco/mujoco.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "g1rtc/planning.hpp"

namespace g1rtc {

namespace {
constexpr double kAlwaysInContactFraction = 0.95;
// An allowed pair may sit inside the clearance band but must not actually interpenetrate.
constexpr double kAllowedPenetrationM = 0.002;

mjModel* copyWithClearance(const mjModel_* m, double clearance_m) {
  mjModel* copy = mj_copyModel(nullptr, m);
  for (int g = 0; g < copy->ngeom; ++g) {
    copy->geom_margin[g] = clearance_m;  // contacts are generated for any pair closer than this
    copy->geom_gap[g] = 0.0;
  }
  return copy;
}
}  // namespace

AllowedCollisionMatrix AllowedCollisionMatrix::sample(const mjModel_* m_in, double clearance_m, int samples, std::uint64_t seed) {
  mjModel* m = copyWithClearance(m_in, clearance_m);
  mjData* d = mj_makeData(m);
  const int nb = static_cast<int>(m->nbody);
  if (m->njnt == 0) throw std::invalid_argument("AllowedCollisionMatrix: the scene has no joints");
  const int robot_root = m->body_rootid[m->jnt_bodyid[0]];
  std::vector<char> robot(static_cast<std::size_t>(nb), 0);
  for (int b = 1; b < nb; ++b) robot[static_cast<std::size_t>(b)] = m->body_rootid[b] == robot_root;
  const auto pair = [nb](int a, int b) { return static_cast<std::size_t>(std::min(a, b) * nb + std::max(a, b)); };
  std::vector<int> contact_count(static_cast<std::size_t>(nb * nb), 0);
  std::vector<char> at_zero_pose(contact_count.size(), 0), seen(contact_count.size(), 0);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int k = 0; k <= samples; ++k) {
    for (int j = 0; j < m->njnt; ++j)
      d->qpos[m->jnt_qposadr[j]] = k == 0 ? 0.0 : m->jnt_range[2 * j] + unit(rng) * (m->jnt_range[2 * j + 1] - m->jnt_range[2 * j]);
    mj_kinematics(m, d);
    mj_collision(m, d);
    std::fill(seen.begin(), seen.end(), 0);
    for (int c = 0; c < d->ncon; ++c) {
      const int b1 = m->geom_bodyid[d->contact[c].geom[0]], b2 = m->geom_bodyid[d->contact[c].geom[1]];
      if (!robot[static_cast<std::size_t>(b1)] || !robot[static_cast<std::size_t>(b2)]) continue;
      const std::size_t i = pair(b1, b2);
      if (seen[i]) continue;
      seen[i] = 1;
      if (k == 0) at_zero_pose[i] = 1;
      else ++contact_count[i];
    }
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  AllowedCollisionMatrix acm;
  acm.nbody_ = nb;
  acm.allowed_.assign(contact_count.size(), 0);
  for (std::size_t i = 0; i < contact_count.size(); ++i) {
    if (at_zero_pose[i] || contact_count[i] >= static_cast<int>(kAlwaysInContactFraction * samples)) {
      acm.allowed_[i] = 1;
      ++acm.allowed_pairs_;
    }
  }
  return acm;
}

bool AllowedCollisionMatrix::allowed(int a, int b) const {
  return allowed_[static_cast<std::size_t>(std::min(a, b) * nbody_ + std::max(a, b))];
}

StateValidityChecker::StateValidityChecker(const mjModel_* m, double clearance_m, AllowedCollisionMatrix acm,
                                           double joint_limit_margin_rad)
    : m_(copyWithClearance(m, clearance_m)), d_(mj_makeData(m_)), acm_(std::move(acm)), joint_limit_margin_rad_(joint_limit_margin_rad) {
  // q is indexed as both qpos and qvel below, which holds for hinge and slide joints only
  if (m_->nq != m_->nv) throw std::invalid_argument("StateValidityChecker: ball and free joints are not supported");
}

StateValidityChecker::~StateValidityChecker() {
  mj_deleteData(d_);
  mj_deleteModel(m_);
}

Validity StateValidityChecker::classify(const VecN& q) {
  ++checks_;
  for (int j = 0; j < m_->njnt; ++j) {
    const double qj = q[m_->jnt_dofadr[j]];
    if (qj < m_->jnt_range[2 * j] + joint_limit_margin_rad_ || qj > m_->jnt_range[2 * j + 1] - joint_limit_margin_rad_)
      return Validity::NearJointLimit;
  }
  for (int i = 0; i < m_->nq; ++i) d_->qpos[i] = q[i];
  mj_kinematics(m_, d_);
  mj_collision(m_, d_);
  for (int c = 0; c < d_->ncon; ++c) {
    const mjContact& con = d_->contact[c];
    const int b1 = m_->geom_bodyid[con.geom[0]], b2 = m_->geom_bodyid[con.geom[1]];
    if (!acm_.allowed(b1, b2) || con.dist < -kAllowedPenetrationM) return Validity::InCollision;
  }
  return Validity::Valid;
}

const char* toString(Validity v) {
  switch (v) {
    case Validity::Valid: return "valid";
    case Validity::NearJointLimit: return "near_joint_limit";
    case Validity::InCollision: return "in_collision";
  }
  return "?";
}

bool StateValidityChecker::segmentValid(const VecN& from, const VecN& to, double resolution_rad) {
  const double span = (to - from).cwiseAbs().maxCoeff();
  const int intervals = std::max(1, static_cast<int>(std::ceil(span / resolution_rad)));
  if (!isValid(to)) return false;
  // visit interior sample i once, at the stride of its largest power-of-two divisor
  int stride = 1;
  while (stride * 2 < intervals) stride *= 2;
  for (; stride >= 1; stride /= 2) {
    for (int i = stride; i < intervals; i += 2 * stride) {
      if (!isValid(VecN(from + (to - from) * (static_cast<double>(i) / intervals)))) return false;
    }
  }
  return true;
}

}  // namespace g1rtc
