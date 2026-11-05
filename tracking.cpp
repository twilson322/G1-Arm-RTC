// Figure-8 tracking with four controllers, with and without an unmodelled payload, and online
// payload estimation (mass only, and mass + COM with a wobbling hand).
#include <cmath>
#include <cstdio>
#include <numbers>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSeed = 7;
const Vec3 kPayloadComTool(0.03, 0.0, -0.02);
constexpr int kLaps = 6;
constexpr int kWarmupLaps = 3;  // estimators converge here; every case is scored on the laps after

struct TrackingCase {
  std::string controller;  // joint_pd_gravity, computed_torque, operational_space, cartesian_impedance
  std::string condition;
  double payload_kg = 0.0;
  bool estimate_payload = false;
  double wobble_rad = 0.0;
};

struct TrackingResult {
  double rms_position_mm = 0, max_position_mm = 0, rms_orientation_deg = 0, peak_torque_ratio = 0;
  long saturated_ticks = 0;
  double payload_estimate_kg = 0;
  Vec3 payload_com_estimate{0, 0, 0};
};

TrackingResult runFigure8(const TrackingCase& c, const std::string& trace_file) {
  StackConfig config = StackConfig::defaultsG1();
  config.detect_collisions = false;
  config.estimate_payload = c.estimate_payload;
  config.adapt_model_to_payload = c.estimate_payload;
  config.payload.estimate_com = c.wobble_rad > 0.0;  // the COM is only identifiable when the hand rotates
  Rig rig(SimConfig{}, kSeed, RobotDescription::homePosture(), config);
  if (c.payload_kg > 0) rig.sim().setPayload(c.payload_kg, kPayloadComTool);
  rig.lockstep->enable(0.5);

  Figure8Task::Params params = figure8At(rig.world.robot, rig.sim().trueQ());
  params.laps = kLaps;
  params.wobble_amplitude_rad = c.wobble_rad;
  if (c.controller == "cartesian_impedance") params.mode = TrackingMode::Impedance;
  auto cartesian = std::make_shared<Figure8Task>(params);
  std::shared_ptr<const Task> task = cartesian;
  if (c.controller == "joint_pd_gravity" || c.controller == "computed_torque") {
    const auto mode = c.controller == "computed_torque" ? TrackingMode::ComputedTorque : TrackingMode::JointHold;
    task = std::make_shared<JointTrajectoryTask>(jointTrajectoryFromTask(*cartesian, rig.world.robot, rig.sim().trueQ()), mode);
  }

  Csv trace(trace_file, "task_time_s,x,y,z,x_ref,y_ref,z_ref,error_mm,payload_estimate_kg,com_x,com_y,com_z");
  TrackingResult result;
  double sum_sq_position = 0, sum_sq_orientation = 0, samples = 0;
  long tick = 0;
  const double score_from_s = params.ramp_s + kWarmupLaps * params.lap_period_s;
  Reference ref;
  ref.q = params.posture;
  const ControlStack& stack = *rig.stack;
  rig.lockstep->execute(task, 60.0, [&](const RobotState&, const TorqueCommand& command) {
    if (stack.supervisor().mode() != Mode::Executing) return;
    const double t = stack.taskTime();
    cartesian->reference(t, ref);
    const rbd::DynamicsCache& dyn = stack.dynamics();
    const double position_error = (dyn.tool_position - ref.tool_position).norm();
    if (t >= score_from_s) {
      sum_sq_position += position_error * position_error;
      sum_sq_orientation += rotationError(ref.tool_rotation, dyn.tool_rotation).squaredNorm();
      result.max_position_mm = std::max(result.max_position_mm, 1e3 * position_error);
      result.peak_torque_ratio = std::max(result.peak_torque_ratio, command.tau.cwiseAbs().cwiseQuotient(stack.model().tau_max).maxCoeff());
      result.saturated_ticks += stack.saturatedJoints() > 0;
      samples += 1;
    }
    if (tick++ % 5 == 0) {
      const Vec3 com = stack.payloadEstimator().comToolFrame();
      trace.row(t, dyn.tool_position.x(), dyn.tool_position.y(), dyn.tool_position.z(), ref.tool_position.x(),
                ref.tool_position.y(), ref.tool_position.z(), 1e3 * position_error, stack.payloadEstimator().massKg(), com.x(),
                com.y(), com.z());
    }
  });
  result.rms_position_mm = samples > 0 ? 1e3 * std::sqrt(sum_sq_position / samples) : NAN;
  result.rms_orientation_deg = samples > 0 ? std::sqrt(sum_sq_orientation / samples) * 180.0 / std::numbers::pi : NAN;
  result.payload_estimate_kg = stack.payloadEstimator().massKg();
  result.payload_com_estimate = stack.payloadEstimator().comToolFrame();
  return result;
}

void tracking() {
  std::printf("== figure-8 tracking (scored on laps %d-%d)\n", kWarmupLaps + 1, kLaps);
  Csv csv("tracking.csv", "controller,condition,rms_pos_mm,max_pos_mm,rms_rot_deg,peak_tau_ratio,sat_ticks,payload_estimate_kg");
  std::vector<TrackingCase> cases;
  for (std::string controller : {"joint_pd_gravity", "computed_torque", "operational_space", "cartesian_impedance"}) {
    cases.push_back({controller, "nominal", 0.0, false});
    cases.push_back({controller, "payload_1kg", 1.0, false});
  }
  for (std::string controller : {"computed_torque", "operational_space"}) {
    cases.push_back({controller, "payload_1kg_adapted", 1.0, true});
    cases.push_back({controller, "nominal_adapted", 0.0, true});  // control: estimator on, no payload
  }
  for (const auto& c : cases) {
    const TrackingResult r = runFigure8(c, "trace_" + c.controller + "_" + c.condition + ".csv");
    csv.row(c.controller, c.condition, r.rms_position_mm, r.max_position_mm, r.rms_orientation_deg, r.peak_torque_ratio,
            r.saturated_ticks, r.payload_estimate_kg);
    std::printf("  %-20s %-20s rms %6.2f mm  max %6.2f mm  rot %5.2f deg  peak tau %.2f  sat %ld  m_hat %.3f\n",
                c.controller.c_str(), c.condition.c_str(), r.rms_position_mm, r.max_position_mm, r.rms_orientation_deg,
                r.peak_torque_ratio, r.saturated_ticks, r.payload_estimate_kg);
  }
}

void payload() {
  std::printf("== payload estimation (1 kg at (3, 0, -2) cm in the tool frame)\n");
  Csv csv("payload.csv", "case,true_mass_kg,estimated_mass_kg,true_com_x,true_com_z,estimated_com_x,estimated_com_y,estimated_com_z,rms_pos_mm");
  for (double wobble : {0.0, 0.35}) {
    const std::string name = wobble > 0 ? "wobble" : "no_wobble";
    const TrackingResult r = runFigure8({"operational_space", name, 1.0, true, wobble}, "payload_trace_" + name + ".csv");
    const Vec3& c = r.payload_com_estimate;
    csv.row(name, 1.0, r.payload_estimate_kg, kPayloadComTool.x(), kPayloadComTool.z(), c.x(), c.y(), c.z(), r.rms_position_mm);
    std::printf("  %-10s mass %.3f kg  COM (%.3f, %.3f, %.3f) m  rms %.2f mm\n", name.c_str(), r.payload_estimate_kg, c.x(), c.y(),
                c.z(), r.rms_position_mm);
  }
}

const Register kTracking("tracking", tracking);
const Register kPayload("payload", payload);

}  // namespace
}  // namespace g1rtc::experiments
