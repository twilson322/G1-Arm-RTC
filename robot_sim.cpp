#include "g1rtc/robot_sim.hpp"

#include <mujoco/mujoco.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace g1rtc {

namespace {
using RowMajor3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
}

int RobotSim::requireId(int type, const char* name) const {
  const int id = mj_name2id(m_, type, name);
  if (id < 0) throw std::runtime_error(std::string("scene is missing required element: ") + name);
  return id;
}

int RobotSim::bodyId(const char* name) const { return requireId(mjOBJ_BODY, name); }

RobotSim::RobotSim(const std::string& xml_path, const SimConfig& cfg, std::uint64_t seed)
    : cfg_(cfg), rng_(seed) {
  char err[512] = "";
  m_ = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!m_) throw std::runtime_error(std::string("mj_loadXML: ") + err);
  d_ = mj_makeData(m_);
  nv_ = static_cast<int>(m_->nv);
  if (!cfg_.joint_friction)
    for (int i = 0; i < nv_; ++i) m_->dof_frictionloss[i] = 0.0;

  // The drive model indexes actuators by DOF, so require exactly one joint motor per DOF.
  actuator_of_dof_.assign(static_cast<std::size_t>(nv_), -1);
  for (int a = 0; a < m_->nu; ++a) {
    if (m_->actuator_trntype[a] != mjTRN_JOINT) throw std::runtime_error("only joint actuators are supported");
    actuator_of_dof_[static_cast<std::size_t>(m_->jnt_dofadr[m_->actuator_trnid[2 * a]])] = a;
  }
  for (int a : actuator_of_dof_)
    if (a < 0) throw std::runtime_error("every DOF needs a torque motor");

  tool_body_ = requireId(mjOBJ_BODY, "tool");
  sponge_geom_ = requireId(mjOBJ_GEOM, "sponge");
  intruder_body_ = requireId(mjOBJ_BODY, "intruder");
  table_body_ = requireId(mjOBJ_BODY, "table");
  ft_force_adr_ = m_->sensor_adr[requireId(mjOBJ_SENSOR, "ft_force")];
  ft_torque_adr_ = m_->sensor_adr[requireId(mjOBJ_SENSOR, "ft_torque")];
  m_->geom_solref[2 * sponge_geom_] = cfg_.sponge_contact_timeconst_s;
  m_->opt.cone = cfg_.elliptic_friction_cones ? mjCONE_ELLIPTIC : mjCONE_PYRAMIDAL;
  base_tool_mass_ = m_->body_mass[tool_body_];
  base_tool_com_ = Vec3(m_->body_ipos[3 * tool_body_], m_->body_ipos[3 * tool_body_ + 1], m_->body_ipos[3 * tool_body_ + 2]);
  table_height0_ = m_->body_pos[3 * table_body_ + 2];

  std::uniform_real_distribution<double> unit_interval(-1.0, 1.0);
  torque_constant_scale_ = VecN::Ones(nv_);
  for (int i = 0; i < nv_; ++i) torque_constant_scale_[i] = 1.0 + cfg_.torque_constant_error * unit_interval(rng_);
  efc_contact_force_.reserve(256);
  joint_scratch_a_.assign(static_cast<std::size_t>(nv_), 0.0);
  joint_scratch_b_.assign(static_cast<std::size_t>(nv_), 0.0);
  reset(VecN::Zero(nv_));
}

RobotSim::~RobotSim() {
  mj_deleteData(d_);
  mj_deleteModel(m_);
}

double RobotSim::physicsDt() const { return m_->opt.timestep; }
double RobotSim::time() const { return d_->time; }

void RobotSim::reset(const VecN& q0) {
  mj_resetData(m_, d_);
  for (int i = 0; i < nv_; ++i) d_->qpos[i] = q0[i];
  mj_forward(m_, d_);
  const double counts_to_rad = 2.0 * std::numbers::pi / cfg_.encoder_counts_per_rev;
  q_quantized_prev_ = VecN(nv_);
  for (int i = 0; i < nv_; ++i) q_quantized_prev_[i] = std::round(q0[i] / counts_to_rad) * counts_to_rad;
  qd_filtered_ = tau_drive_ = tau_external_true_ = VecN::Zero(nv_);
  commands_in_flight_.clear();
  active_command_ = TorqueCommand{VecN::Zero(nv_), 0, false};
  have_command_ = false;
  fault_ = true;  // drives power up disabled; the first command must carry reset_fault
  last_command_time_ = -1.0;
  disturbances_.clear();
  state_ = RobotState{};
  state_.q = q_quantized_prev_;
  state_.qd = qd_filtered_;
  state_.applied_torque_impulse = VecN::Zero(nv_);
  updateSensors();
}

void RobotSim::command(const TorqueCommand& cmd) {
  // half a step of slack so a command due exactly on a step boundary is not delayed by one more
  const double reaches_motor = d_->time + (cfg_.command_delay_physics_steps - 0.5) * m_->opt.timestep;
  commands_in_flight_.emplace_back(reaches_motor, cmd);
}

void RobotSim::setPayload(double mass_kg, const Vec3& com_tool_m) {
  const double total = base_tool_mass_ + mass_kg;
  const Vec3 com = (base_tool_mass_ * base_tool_com_ + mass_kg * com_tool_m) / total;
  m_->body_mass[tool_body_] = total;
  for (int i = 0; i < 3; ++i) m_->body_ipos[3 * tool_body_ + i] = com[i];
  m_->body_sameframe[tool_body_] = 0;  // the compiler flagged COM == origin; MuJoCo would skip the offset
}

void RobotSim::setIntruderPose(const Vec3& position, const Mat3& rotation) {
  const int mocap = m_->body_mocapid[intruder_body_];
  const Eigen::Quaterniond q(rotation);
  for (int i = 0; i < 3; ++i) d_->mocap_pos[3 * mocap + i] = position[i];
  const double wxyz[4] = {q.w(), q.x(), q.y(), q.z()};
  for (int i = 0; i < 4; ++i) d_->mocap_quat[4 * mocap + i] = wxyz[i];
}

double RobotSim::contactForceWith(const char* env_geom) const {
  const int g = requireId(mjOBJ_GEOM, env_geom);
  double total = 0.0;
  for (int c = 0; c < d_->ncon; ++c) {
    const mjContact& con = d_->contact[c];
    if (con.efc_address < 0 || (con.geom[0] != g && con.geom[1] != g)) continue;
    double f6[6];
    mj_contactForce(m_, d_, c, f6);
    total += Vec3(f6[0], f6[1], f6[2]).norm();
  }
  return total;
}

void RobotSim::setTableHeightOffset(double offset_m) {
  m_->body_pos[3 * table_body_ + 2] = table_height0_ + offset_m;
  mj_forward(m_, d_);
}

VecN RobotSim::trueQ() const { return Eigen::Map<const Eigen::VectorXd>(d_->qpos, nv_); }
VecN RobotSim::trueQd() const { return Eigen::Map<const Eigen::VectorXd>(d_->qvel, nv_); }

void RobotSim::step() {
  const double t = d_->time;
  applyDriveModel(t);
  applyDisturbances(t);
  mj_step(m_, d_);
  state_.applied_torque_impulse += tau_drive_ * m_->opt.timestep;
  updateGroundTruth();
  updateSensors();
}

void RobotSim::applyDriveModel(double t) {
  while (!commands_in_flight_.empty() && commands_in_flight_.front().first <= t) {
    active_command_ = std::move(commands_in_flight_.front().second);
    commands_in_flight_.pop_front();
    have_command_ = true;
    last_command_time_ = t;
    if (active_command_.reset_fault) fault_ = false;
  }
  if (have_command_ && t - last_command_time_ > cfg_.watchdog_timeout_s) fault_ = true;
  for (int i = 0; i < nv_; ++i) {
    const int a = actuator_of_dof_[static_cast<std::size_t>(i)];
    const double current_limit_as_torque = m_->actuator_ctrlrange[2 * a + 1];
    const double requested = (fault_ || !have_command_) ? -cfg_.fault_damping_nms_per_rad * d_->qvel[i]
                                                        : active_command_.tau[i];
    tau_drive_[i] = std::clamp(requested, -current_limit_as_torque, current_limit_as_torque);
    d_->ctrl[a] = torque_constant_scale_[i] * tau_drive_[i];  // ctrlrange clamps again: harmless
  }
}

void RobotSim::applyDisturbances(double t) {
  for (const auto& dist : disturbances_)
    for (int k = 0; k < 3; ++k) d_->xfrc_applied[6 * dist.body + k] = 0.0;
  for (const auto& dist : disturbances_) {
    if (t >= dist.start_s && t < dist.start_s + dist.duration_s)
      for (int k = 0; k < 3; ++k) d_->xfrc_applied[6 * dist.body + k] += dist.force_world_n[k];
  }
}

void RobotSim::updateGroundTruth() {
  // Contact rows of the constraint force, mapped to joint torques. Joint friction and joint-limit
  // rows are left out: those are internal to the robot, not something a collision detector
  // should report.
  const int nefc = d_->nefc;
  efc_contact_force_.assign(static_cast<std::size_t>(nefc), 0.0);
  for (int i = 0; i < nefc; ++i)
    if (d_->efc_type[i] >= mjCNSTR_CONTACT_FRICTIONLESS) efc_contact_force_[static_cast<std::size_t>(i)] = d_->efc_force[i];
  std::fill(joint_scratch_a_.begin(), joint_scratch_a_.end(), 0.0);
  if (nefc > 0) mj_mulJacTVec(m_, d_, joint_scratch_a_.data(), efc_contact_force_.data());
  std::fill(joint_scratch_b_.begin(), joint_scratch_b_.end(), 0.0);
  for (const auto& dist : disturbances_) {
    const double* f = d_->xfrc_applied + 6 * dist.body;
    if (f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0) continue;
    mj_applyFT(m_, d_, f, f + 3, d_->xipos + 3 * dist.body, dist.body, joint_scratch_b_.data());
  }
  for (int i = 0; i < nv_; ++i)
    tau_external_true_[i] = joint_scratch_a_[static_cast<std::size_t>(i)] + joint_scratch_b_[static_cast<std::size_t>(i)];

  tool_contact_force_.setZero();
  for (int c = 0; c < d_->ncon; ++c) {
    const mjContact& con = d_->contact[c];
    if (con.efc_address < 0 || (con.geom[0] != sponge_geom_ && con.geom[1] != sponge_geom_)) continue;
    double f6[6];
    mj_contactForce(m_, d_, c, f6);
    // mj_contactForce gives the force of geom[0] on geom[1] in the contact frame
    const Vec3 on_geom1 = Eigen::Map<const RowMajor3>(con.frame).transpose() * Vec3(f6[0], f6[1], f6[2]);
    tool_contact_force_ += con.geom[1] == sponge_geom_ ? on_geom1 : Vec3(-on_geom1);
  }
}

void RobotSim::updateSensors() {
  const double counts_to_rad = 2.0 * std::numbers::pi / cfg_.encoder_counts_per_rev;
  const double dt = m_->opt.timestep;
  const double cutoff_hz = cfg_.velocity_filter_cutoff_hz;
  const double filter_gain = cutoff_hz > 0.0 ? 1.0 - std::exp(-2.0 * std::numbers::pi * cutoff_hz * dt) : 1.0;
  for (int i = 0; i < nv_; ++i) {
    const double q_quantized = std::round(d_->qpos[i] / counts_to_rad) * counts_to_rad;
    const double qd_differenced = (q_quantized - q_quantized_prev_[i]) / dt;
    qd_filtered_[i] += filter_gain * (qd_differenced - qd_filtered_[i]);
    q_quantized_prev_[i] = q_quantized;
    state_.q[i] = q_quantized;
    state_.qd[i] = qd_filtered_[i];
  }
  state_.time_s = d_->time;
  for (int k = 0; k < 3; ++k) {
    state_.wrist_ft_reading[k] = d_->sensordata[ft_force_adr_ + k] + cfg_.ft_force_noise_std_n * unit_normal_(rng_);
    state_.wrist_ft_reading[3 + k] = d_->sensordata[ft_torque_adr_ + k] + cfg_.ft_torque_noise_std_nm * unit_normal_(rng_);
  }
  state_.drive_fault = fault_;
  state_.last_command_seq = active_command_.seq;
}

}  // namespace g1rtc
