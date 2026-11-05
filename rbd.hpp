// Rigid-body dynamics for a fixed-base kinematic tree of hinge joints, written from scratch.
//
// The controllers never call MuJoCo. On a real robot the simulator is not there, so the
// controller carries its own model (built once from the MJCF, then free to diverge from the plant,
// e.g. by an estimated payload). MuJoCo is the ground truth the tests compare against.
//
// Every quantity is in the world frame. Per body: rotation and origin, the hinge axis, the hinge
// anchor (a point on both parent and child, so the parent's acceleration of it is the child's
// too), and the COM. With those the recursions below are the textbook ones without spatial-vector
// algebra: forward kinematics, recursive Newton-Euler (RNEA), the composite-rigid-body mass matrix
// (CRBA), point Jacobians and J-dot * q-dot. Notation in formulas follows the usual robotics
// symbols: q, qd, qdd joint position, velocity, acceleration; M mass matrix; J Jacobian; g gravity
// torque; h = C(q, qd) qd + g.
//
// The algorithms are templates on the scalar so they also run on Dual numbers (dual.hpp), which
// gives exact time derivatives for the momentum observer.
//
// Real-time note: vectors indexed by DOF use Eigen's "dynamic size with a compile-time maximum"
// (kMaxDof), so they live on the stack and nothing here allocates once the scratch structures are
// sized. tests/test_rt.cpp checks the whole control tick with a counting operator new.
#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector>

#include "g1rtc/dual.hpp"

struct mjModel_;

namespace g1rtc {

inline constexpr int kMaxDof = 16;
template <class S> using Vec3T = Eigen::Matrix<S, 3, 1>;
template <class S> using Mat3T = Eigen::Matrix<S, 3, 3>;
template <class S> using VecNT = Eigen::Matrix<S, Eigen::Dynamic, 1, 0, kMaxDof, 1>;
template <class S> using MatNT = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic, 0, kMaxDof, kMaxDof>;
template <class S> using Mat6NT = Eigen::Matrix<S, 6, Eigen::Dynamic, 0, 6, kMaxDof>;
using Vec3 = Vec3T<double>;
using Mat3 = Mat3T<double>;
using VecN = VecNT<double>;
using MatN = MatNT<double>;
using Mat6N = Mat6NT<double>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat3N = Eigen::Matrix<double, 3, Eigen::Dynamic, 0, 3, kMaxDof>;
using MatN6 = Eigen::Matrix<double, Eigen::Dynamic, 6, 0, kMaxDof, 6>;

inline Mat3 skew(const Vec3& v) {
  Mat3 s;
  s << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return s;
}

/// Orientation error log(R_desired R^T) as a rotation vector in the world frame, rad.
Vec3 rotationError(const Mat3& R_desired, const Mat3& R);

namespace rbd {

struct Body {
  std::string name;
  int parent = -1;
  int dof = -1;  // -1: welded to its parent
  Vec3 origin_in_parent{0, 0, 0};  // at q = 0
  Mat3 rotation_in_parent = Mat3::Identity();
  Vec3 joint_axis_local{0, 0, 1};
  Vec3 joint_anchor_local{0, 0, 0};
  double mass = 0.0;
  Vec3 com_local{0, 0, 0};
  Mat3 inertia_about_com_local = Mat3::Zero();
};

/// A frame rigidly attached to a body (a MuJoCo site).
struct Frame {
  int body = 0;
  Vec3 offset_local{0, 0, 0};
  Mat3 rotation_local = Mat3::Identity();
};

struct Model {
  std::vector<Body> bodies;  // bodies[0] is the world; parents precede children
  int nv = 0;
  std::vector<int> body_of_dof;
  std::vector<std::string> joint_names;
  VecN armature, q_min, q_max, tau_max, qd_max;
  Vec3 gravity{0, 0, -9.81};

  static Model fromMujoco(const mjModel_* m);
  static Model fromXml(const std::string& path);
  int bodyId(const std::string& name) const;
  Frame site(const mjModel_* m, const std::string& name) const;
  /// Rigidly attach a payload (COM in the body frame, inertia about its own COM). Used to give the
  /// controller model an estimated payload.
  void addPayload(int body, double mass, const Vec3& com_local, const Mat3& inertia_local = Mat3::Zero());
  double totalMass() const;
};

template <class S>
struct Kinematics {
  std::vector<Mat3T<S>> rotation;
  std::vector<Vec3T<S>> origin, joint_axis, joint_anchor, com;
  explicit Kinematics(const Model& m = Model{}) { resize(m); }
  void resize(const Model& m) {
    const auto n = m.bodies.size();
    rotation.assign(n, Mat3T<S>::Identity());
    origin.assign(n, Vec3T<S>::Zero());
    joint_axis = joint_anchor = com = origin;
  }
};

/// Scratch for RNEA and CRBA, sized once per model.
template <class S>
struct Workspace {
  std::vector<Vec3T<S>> angular_velocity, angular_acceleration, com_acceleration;
  std::vector<Vec3T<S>> force, moment_about_anchor;  // what the parent exerts on the subtree
  std::vector<S> subtree_mass;
  std::vector<Vec3T<S>> subtree_mass_moment;           // sum m_i c_i
  std::vector<Mat3T<S>> subtree_inertia_about_origin;  // about the world origin
  explicit Workspace(const Model& m = Model{}) { resize(m); }
  void resize(const Model& m) {
    const auto n = m.bodies.size();
    angular_velocity.assign(n, Vec3T<S>::Zero());
    angular_acceleration = com_acceleration = force = moment_about_anchor = subtree_mass_moment = angular_velocity;
    subtree_mass.assign(n, S(0.0));
    subtree_inertia_about_origin.assign(n, Mat3T<S>::Zero());
  }
};

template <class S>
Mat3T<S> axisAngle(const Vec3& axis, const S& angle) {
  using std::cos;
  using std::sin;
  const Mat3 K = skew(axis);
  return Mat3T<S>::Identity() + K.cast<S>() * sin(angle) + (K * K).cast<S>() * (S(1.0) - cos(angle));
}

template <class S, class Q>
void forwardKinematics(const Model& m, const Q& q, Kinematics<S>& k) {
  for (std::size_t i = 1; i < m.bodies.size(); ++i) {
    const Body& b = m.bodies[i];
    const auto parent = static_cast<std::size_t>(b.parent);
    Mat3T<S> rotation_in_parent = b.rotation_in_parent.cast<S>();
    Vec3T<S> origin_in_parent = b.origin_in_parent.cast<S>();
    if (b.dof >= 0) {
      // rotation about the anchor: the anchor stays put, the body origin swings around it
      const Mat3T<S> hinge = axisAngle<S>(b.joint_axis_local, S(q[b.dof]));
      const Vec3T<S> anchor = b.joint_anchor_local.cast<S>();
      origin_in_parent += rotation_in_parent * (anchor - hinge * anchor);
      rotation_in_parent = rotation_in_parent * hinge;
    }
    k.rotation[i] = k.rotation[parent] * rotation_in_parent;
    k.origin[i] = k.origin[parent] + k.rotation[parent] * origin_in_parent;
    k.joint_axis[i] = b.dof >= 0 ? Vec3T<S>(k.rotation[i] * b.joint_axis_local.cast<S>()) : Vec3T<S>::Zero();
    k.joint_anchor[i] = k.origin[i] + k.rotation[i] * b.joint_anchor_local.cast<S>();
    k.com[i] = k.origin[i] + k.rotation[i] * b.com_local.cast<S>();
  }
}

/// Velocity and acceleration sweep from the root, shared by RNEA and J-dot * q-dot. With
/// `include_gravity` the base accelerates upward at g, which adds the gravity load to every body.
template <class S, class V1, class V2>
void propagateMotion(const Model& m, const Kinematics<S>& k, const V1& qd, const V2& qdd, bool include_gravity,
                     Workspace<S>& w) {
  w.angular_velocity[0].setZero();
  w.angular_acceleration[0].setZero();
  w.com_acceleration[0] = include_gravity ? Vec3T<S>((-m.gravity).cast<S>()) : Vec3T<S>::Zero();
  for (std::size_t i = 1; i < m.bodies.size(); ++i) {
    const Body& b = m.bodies[i];
    const auto parent = static_cast<std::size_t>(b.parent);
    const Vec3T<S>& omega_parent = w.angular_velocity[parent];
    const Vec3T<S>& alpha_parent = w.angular_acceleration[parent];
    if (b.dof >= 0) {
      const S qd_i = S(qd[b.dof]);
      const S qdd_i = S(qdd[b.dof]);
      w.angular_velocity[i] = omega_parent + k.joint_axis[i] * qd_i;
      w.angular_acceleration[i] = alpha_parent + k.joint_axis[i] * qdd_i + omega_parent.cross(k.joint_axis[i]) * qd_i;
    } else {
      w.angular_velocity[i] = omega_parent;
      w.angular_acceleration[i] = alpha_parent;
    }
    const Vec3T<S> parent_com_to_anchor = k.joint_anchor[i] - k.com[parent];
    const Vec3T<S> anchor_acceleration = w.com_acceleration[parent] + alpha_parent.cross(parent_com_to_anchor) +
                                         omega_parent.cross(omega_parent.cross(parent_com_to_anchor));
    const Vec3T<S> anchor_to_com = k.com[i] - k.joint_anchor[i];
    w.com_acceleration[i] = anchor_acceleration + w.angular_acceleration[i].cross(anchor_to_com) +
                            w.angular_velocity[i].cross(w.angular_velocity[i].cross(anchor_to_com));
  }
}

/// Inverse dynamics tau = M(q) qdd + C(q, qd) qd + g(q) (gravity optional), armature included.
template <class S, class V1, class V2>
void rnea(const Model& m, const Kinematics<S>& k, const V1& qd, const V2& qdd, bool include_gravity, Workspace<S>& w,
          VecNT<S>& tau) {
  propagateMotion(m, k, qd, qdd, include_gravity, w);
  const std::size_t nb = m.bodies.size();
  for (std::size_t i = 1; i < nb; ++i) {
    const Body& b = m.bodies[i];
    const Mat3T<S> inertia_world = k.rotation[i] * b.inertia_about_com_local.cast<S>() * k.rotation[i].transpose();
    const Vec3T<S> f = w.com_acceleration[i] * S(b.mass);
    w.force[i] = f;
    w.moment_about_anchor[i] = inertia_world * w.angular_acceleration[i] +
                               w.angular_velocity[i].cross(inertia_world * w.angular_velocity[i]) +
                               (k.com[i] - k.joint_anchor[i]).cross(f);
  }
  tau.resize(m.nv);
  for (std::size_t i = nb - 1; i >= 1; --i) {
    const Body& b = m.bodies[i];
    if (b.dof >= 0) tau[b.dof] = k.joint_axis[i].dot(w.moment_about_anchor[i]) + S(m.armature[b.dof]) * S(qdd[b.dof]);
    const auto parent = static_cast<std::size_t>(b.parent);
    if (parent > 0) {
      w.force[parent] += w.force[i];
      w.moment_about_anchor[parent] += w.moment_about_anchor[i] + (k.joint_anchor[i] - k.joint_anchor[parent]).cross(w.force[i]);
    }
  }
}

/// Joint-space mass matrix by the composite-rigid-body algorithm (armature on the diagonal).
template <class S>
void crba(const Model& m, const Kinematics<S>& k, Workspace<S>& w, MatNT<S>& M) {
  const std::size_t nb = m.bodies.size();
  for (std::size_t i = 1; i < nb; ++i) {
    const Body& b = m.bodies[i];
    const S mass = S(b.mass);
    const Vec3T<S>& c = k.com[i];
    w.subtree_mass[i] = mass;
    w.subtree_mass_moment[i] = c * mass;
    w.subtree_inertia_about_origin[i] = k.rotation[i] * b.inertia_about_com_local.cast<S>() * k.rotation[i].transpose() +
                                        (Mat3T<S>::Identity() * c.squaredNorm() - c * c.transpose()) * mass;
  }
  for (std::size_t i = nb - 1; i >= 1; --i) {
    const auto parent = static_cast<std::size_t>(m.bodies[i].parent);
    if (parent > 0) {
      w.subtree_mass[parent] += w.subtree_mass[i];
      w.subtree_mass_moment[parent] += w.subtree_mass_moment[i];
      w.subtree_inertia_about_origin[parent] += w.subtree_inertia_about_origin[i];
    }
  }
  M.setZero(m.nv, m.nv);
  for (std::size_t j = 1; j < nb; ++j) {
    const int dof_j = m.bodies[j].dof;
    if (dof_j < 0) continue;
    const Vec3T<S>& axis = k.joint_axis[j];
    const Vec3T<S>& anchor = k.joint_anchor[j];
    // wrench that gives the subtree of j a unit acceleration about joint j, starting from rest
    const Vec3T<S> force = axis.cross(w.subtree_mass_moment[j] - anchor * w.subtree_mass[j]);
    const Vec3T<S> moment_about_origin =
        w.subtree_inertia_about_origin[j] * axis - w.subtree_mass_moment[j].cross(axis.cross(anchor));
    for (int a = static_cast<int>(j); a > 0; a = m.bodies[static_cast<std::size_t>(a)].parent) {
      const auto ancestor = static_cast<std::size_t>(a);
      const int dof_a = m.bodies[ancestor].dof;
      if (dof_a < 0) continue;
      const S entry = k.joint_axis[ancestor].dot(moment_about_origin - k.joint_anchor[ancestor].cross(force));
      M(dof_a, dof_j) = entry;
      M(dof_j, dof_a) = entry;
    }
    M(dof_j, dof_j) += S(m.armature[dof_j]);
  }
}

/// 6 x nv Jacobian of a world point attached to `body`: rows 0-2 its linear velocity, 3-5 the
/// body's angular velocity.
template <class S>
void pointJacobian(const Model& m, const Kinematics<S>& k, int body, const Vec3T<S>& point, Mat6NT<S>& J) {
  J.setZero(6, m.nv);
  for (int a = body; a > 0; a = m.bodies[static_cast<std::size_t>(a)].parent) {
    const auto ancestor = static_cast<std::size_t>(a);
    const int dof = m.bodies[ancestor].dof;
    if (dof < 0) continue;
    J.template block<3, 1>(0, dof) = k.joint_axis[ancestor].cross(point - k.joint_anchor[ancestor]);
    J.template block<3, 1>(3, dof) = k.joint_axis[ancestor];
  }
}

/// J-dot * q-dot for the same point: its acceleration (and the body's angular acceleration) when
/// qdd = 0 and gravity is off.
template <class S, class V>
Eigen::Matrix<S, 6, 1> pointJdotQd(const Model& m, const Kinematics<S>& k, int body, const Vec3T<S>& point, const V& qd,
                                   Workspace<S>& w) {
  const VecNT<S> zero = VecNT<S>::Zero(m.nv);
  propagateMotion(m, k, qd, zero, false, w);
  const auto b = static_cast<std::size_t>(body);
  const Vec3T<S> com_to_point = point - k.com[b];
  Eigen::Matrix<S, 6, 1> out;
  out.template head<3>() = w.com_acceleration[b] + w.angular_acceleration[b].cross(com_to_point) +
                           w.angular_velocity[b].cross(w.angular_velocity[b].cross(com_to_point));
  out.template tail<3>() = w.angular_acceleration[b];
  return out;
}

/// Everything the controllers need at one sample, computed once per tick and shared.
struct DynamicsCache {
  Kinematics<double> kin;
  Workspace<double> work;
  MatN M;
  Eigen::LLT<MatN> M_cholesky;
  VecN g, h;  // gravity torque; full bias C(q, qd) qd + g
  Mat6N J;    // tool frame Jacobian
  Vec6 Jdqd;  // J-dot * q-dot of the tool frame
  Vec3 tool_position;
  Mat3 tool_rotation;
  explicit DynamicsCache(const Model& m = Model{}) : kin(m), work(m) {}
  void update(const Model& m, const Frame& tool, const VecN& q, const VecN& qd);
};

/// World pose of a frame in the current kinematics.
inline void framePose(const Kinematics<double>& k, const Frame& f, Vec3& position, Mat3& rotation) {
  const auto b = static_cast<std::size_t>(f.body);
  rotation = k.rotation[b] * f.rotation_local;
  position = k.origin[b] + k.rotation[b] * f.offset_local;
}

/// beta = g(q) - C(q, qd)^T qd, the momentum-observer bias term. With Mdot = C + C^T,
/// C^T qd = Mdot qd - C qd, and Mdot qd is the dual part of RNEA(q + eps qd, 0, qd) without
/// gravity: O(n) and exact, against O(n^2) and approximate for finite differences of M.
VecN observerBias(const Model& m, const VecN& q, const VecN& qd, Kinematics<double>& kin, Workspace<double>& work,
                  Kinematics<Dual>& kin_dual, Workspace<Dual>& work_dual);

}  // namespace rbd
}  // namespace g1rtc
