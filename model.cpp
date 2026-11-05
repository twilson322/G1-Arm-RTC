#include <mujoco/mujoco.h>

#include <Eigen/Geometry>
#include <limits>
#include <stdexcept>

#include "g1rtc/rbd.hpp"
#include "g1rtc/robot_description.hpp"

namespace g1rtc {

Vec3 rotationError(const Mat3& R_desired, const Mat3& R) {
  const Eigen::AngleAxisd aa(Mat3(R_desired * R.transpose()));
  return aa.axis() * aa.angle();
}

namespace rbd {

namespace {
Mat3 quatToRotation(const double* wxyz) { return Eigen::Quaterniond(wxyz[0], wxyz[1], wxyz[2], wxyz[3]).toRotationMatrix(); }
Vec3 vec3(const double* p) { return {p[0], p[1], p[2]}; }
}  // namespace

Model Model::fromMujoco(const mjModel_* mm) {
  const mjModel* m = mm;
  Model out;
  out.nv = static_cast<int>(m->nv);
  if (out.nv > kMaxDof) throw std::runtime_error("model has more DOF than kMaxDof");
  out.gravity = vec3(m->opt.gravity);
  out.bodies.resize(static_cast<std::size_t>(m->nbody));
  out.body_of_dof.assign(static_cast<std::size_t>(m->nv), -1);
  out.joint_names.resize(static_cast<std::size_t>(m->nv));
  out.armature = VecN::Zero(m->nv);
  out.q_min = VecN::Zero(m->nv);
  out.q_max = VecN::Zero(m->nv);
  out.tau_max = VecN::Constant(m->nv, std::numeric_limits<double>::infinity());
  // The MJCF has no velocity limits. 8 rad/s is the supervisor's speed trip, set above anything
  // the planners command (3-6 rad/s) and well below the G1 actuators' no-load speeds.
  out.qd_max = VecN::Constant(m->nv, 8.0);
  out.bodies[0].name = "world";
  for (int i = 1; i < m->nbody; ++i) {
    Body& b = out.bodies[static_cast<std::size_t>(i)];
    const char* name = mj_id2name(m, mjOBJ_BODY, i);
    b.name = name ? name : "";
    b.parent = m->body_parentid[i];
    b.origin_in_parent = vec3(m->body_pos + 3 * i);
    b.rotation_in_parent = quatToRotation(m->body_quat + 4 * i);
    b.mass = m->body_mass[i];
    b.com_local = vec3(m->body_ipos + 3 * i);
    const Mat3 principal_axes = quatToRotation(m->body_iquat + 4 * i);
    b.inertia_about_com_local = principal_axes * vec3(m->body_inertia + 3 * i).asDiagonal() * principal_axes.transpose();
    if (m->body_jntnum[i] > 1) throw std::runtime_error("body with more than one joint: " + b.name);
    if (m->body_jntnum[i] == 1) {
      const int j = m->body_jntadr[i];
      if (m->jnt_type[j] != mjJNT_HINGE) throw std::runtime_error("only hinge joints are supported: " + b.name);
      if (m->qpos0[m->jnt_qposadr[j]] != 0.0) throw std::runtime_error("nonzero joint reference: " + b.name);
      const int dof = m->jnt_dofadr[j];
      b.dof = dof;
      b.joint_axis_local = vec3(m->jnt_axis + 3 * j).normalized();
      b.joint_anchor_local = vec3(m->jnt_pos + 3 * j);
      out.body_of_dof[static_cast<std::size_t>(dof)] = i;
      const char* joint_name = mj_id2name(m, mjOBJ_JOINT, j);
      out.joint_names[static_cast<std::size_t>(dof)] = joint_name ? joint_name : "";
      out.armature[dof] = m->dof_armature[dof];
      out.q_min[dof] = m->jnt_range[2 * j];
      out.q_max[dof] = m->jnt_range[2 * j + 1];
    }
  }
  for (int a = 0; a < m->nu; ++a) {
    if (m->actuator_trntype[a] != mjTRN_JOINT) continue;
    const int dof = m->jnt_dofadr[m->actuator_trnid[2 * a]];
    out.tau_max[dof] = m->actuator_ctrlrange[2 * a + 1];
  }
  return out;
}

Model Model::fromXml(const std::string& path) {
  char err[512] = "";
  mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("mj_loadXML: " + std::string(err));
  Model out = fromMujoco(m);
  mj_deleteModel(m);
  return out;
}

int Model::bodyId(const std::string& name) const {
  for (std::size_t i = 0; i < bodies.size(); ++i)
    if (bodies[i].name == name) return static_cast<int>(i);
  throw std::runtime_error("no body named " + name);
}

Frame Model::site(const mjModel_* mm, const std::string& name) const {
  const mjModel* m = mm;
  const int s = mj_name2id(m, mjOBJ_SITE, name.c_str());
  if (s < 0) throw std::runtime_error("no site named " + name);
  Frame f;
  f.body = m->site_bodyid[s];
  f.offset_local = vec3(m->site_pos + 3 * s);
  f.rotation_local = quatToRotation(m->site_quat + 4 * s);
  return f;
}

void Model::addPayload(int body, double payload_mass, const Vec3& payload_com, const Mat3& payload_inertia) {
  Body& b = bodies[static_cast<std::size_t>(body)];
  const double total = b.mass + payload_mass;
  if (total <= 0.0) return;
  const Vec3 com = (b.mass * b.com_local + payload_mass * payload_com) / total;
  auto parallelAxis = [](double mass, const Vec3& d) { return mass * (d.squaredNorm() * Mat3::Identity() - d * d.transpose()); };
  b.inertia_about_com_local = b.inertia_about_com_local + parallelAxis(b.mass, b.com_local - com) + payload_inertia +
                              parallelAxis(payload_mass, payload_com - com);
  b.mass = total;
  b.com_local = com;
}

double Model::totalMass() const {
  double total = 0;
  for (const auto& b : bodies) total += b.mass;
  return total;
}

void DynamicsCache::update(const Model& m, const Frame& tool, const VecN& q, const VecN& qd) {
  forwardKinematics(m, q, kin);
  crba(m, kin, work, M);
  M_cholesky.compute(M);
  const VecN zero = VecN::Zero(m.nv);
  rnea(m, kin, qd, zero, true, work, h);
  rnea(m, kin, zero, zero, true, work, g);
  framePose(kin, tool, tool_position, tool_rotation);
  pointJacobian(m, kin, tool.body, tool_position, J);
  Jdqd = pointJdotQd(m, kin, tool.body, tool_position, qd, work);
}

VecN observerBias(const Model& m, const VecN& q, const VecN& qd, Kinematics<double>& kin, Workspace<double>& work,
                  Kinematics<Dual>& kin_dual, Workspace<Dual>& work_dual) {
  const VecN zero = VecN::Zero(m.nv);
  forwardKinematics(m, q, kin);
  VecN coriolis, gravity;
  rnea(m, kin, qd, zero, false, work, coriolis);
  rnea(m, kin, zero, zero, true, work, gravity);
  VecNT<Dual> q_along_qd(m.nv), zero_dual(m.nv), qd_as_acceleration(m.nv);
  for (int i = 0; i < m.nv; ++i) {
    q_along_qd[i] = Dual(q[i], qd[i]);
    zero_dual[i] = Dual(0.0);
    qd_as_acceleration[i] = Dual(qd[i]);
  }
  forwardKinematics(m, q_along_qd, kin_dual);
  VecNT<Dual> M_times_qd;
  rnea(m, kin_dual, zero_dual, qd_as_acceleration, false, work_dual, M_times_qd);
  VecN beta(m.nv);
  for (int i = 0; i < m.nv; ++i) beta[i] = gravity[i] - (M_times_qd[i].b - coriolis[i]);  // g - C^T qd
  return beta;
}

}  // namespace rbd

RobotDescription RobotDescription::fromMujoco(const mjModel_* m) {
  RobotDescription d;
  d.model = rbd::Model::fromMujoco(m);
  d.tool = d.model.site(m, "tool_tip");
  d.ft_sensor = d.model.site(m, "ft_site");
  if (d.model.nv != static_cast<int>(homePosture().size()))
    throw std::runtime_error("scene does not have the 10 DOF the G1 upper-body defaults assume");
  return d;
}

VecN RobotDescription::homePosture() {
  // waist yaw/roll/pitch, right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (rad):
  // forearm forward and slightly down, hand in front of the table, clear of the body and limits
  VecN q(10);
  q << 0, 0, 0, -0.5, -0.15, 0.0, 0.8, 0, 0, 0;
  return q;
}

}  // namespace g1rtc
