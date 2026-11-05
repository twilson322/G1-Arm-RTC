// Force-controlled wiping on a table whose height is uncertain by +/-10 mm.
#include <cmath>
#include <cstdio>
#include <numbers>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSeed = 3;
const Vec3 kSurfacePoint(0.40, -0.22, 0.65);  // sponge centre touching the nominal table
constexpr double kStartHeightAboveSurfaceM = 0.03;
constexpr double kForceRampS = 1.5;
constexpr double kNoContactForceN = 0.5;

/// Baseline without force control: the wipe circle, commanded `depth_m` below the nominal
/// surface, descending there from the start height over `descent_s` with a cosine blend.
class PressIntoSurfaceTask final : public Task {
 public:
  PressIntoSurfaceTask(std::shared_ptr<const WipeTask> wipe, double depth_m, double descent_s)
      : wipe_(std::move(wipe)), depth_m_(depth_m), descent_s_(descent_s) {}
  void reference(double t, Reference& r) const override {
    wipe_->reference(t, r);
    const double drop = kStartHeightAboveSurfaceM + depth_m_;
    const double phase = std::numbers::pi * std::min(1.0, t / descent_s_);
    const bool descending = t < descent_s_;
    r.tool_position.z() += kStartHeightAboveSurfaceM - drop * 0.5 * (1 - std::cos(phase));
    r.tool_velocity.z() = descending ? -drop * 0.5 * std::numbers::pi / descent_s_ * std::sin(phase) : 0.0;
    r.tool_acceleration.z() = descending ? -drop * 0.5 * std::pow(std::numbers::pi / descent_s_, 2) * std::cos(phase) : 0.0;
  }
  double duration() const override { return wipe_->duration(); }
  TrackingMode mode() const override { return TrackingMode::OperationalSpace; }
  const char* name() const override { return "press_into_surface"; }

 private:
  std::shared_ptr<const WipeTask> wipe_;
  double depth_m_, descent_s_;
};

struct WipeController {
  std::string name;
  bool force_feedback, friction_compensation, position_only, detect_collisions;
};

void wipe() {
  std::printf("== force-controlled wiping, 10 N, table height off by -10/0/+10 mm\n");
  Csv csv("wipe.csv", "controller,table_offset_mm,mean_force,rms_force_err,min_force,max_force,contact_loss_pct,rms_tangential_mm,collision_trips");
  const World probe(SimConfig{}, kSeed, RobotDescription::homePosture());
  Vec3 home_position;
  Mat3 home_rotation;
  toolPose(probe.robot, RobotDescription::homePosture(), home_position, home_rotation);
  const IkResult start = solveIk(probe.robot.model, probe.robot.tool, kSurfacePoint + Vec3(0, 0, kStartHeightAboveSurfaceM),
                                 home_rotation, RobotDescription::homePosture(), RobotDescription::homePosture());
  if (!start.converged) throw std::runtime_error("wipe start pose unreachable");

  // Detection runs with the deployed controllers (force PI). The baselines run without it so their
  // statistics cover the whole task: pressing with a position task is free motion running into
  // the table, which the detector correctly stops, and the feed-forward-only variant's larger force
  // transients trip thresholds calibrated on the PI controller. The *_detect row counts those trips.
  const std::vector<WipeController> controllers = {
      {"hybrid_pi", true, false, false, true},
      {"hybrid_pi_friction_ff", true, true, false, true},
      {"hybrid_feedforward", false, false, false, false},
      {"osc_position", false, false, true, false},
      {"hybrid_feedforward_detect", false, false, false, true},
  };
  for (double offset_m : {-0.01, 0.0, 0.01}) {
    for (const WipeController& controller : controllers) {
      StackConfig config = StackConfig::defaultsG1();
      config.hybrid.use_force_feedback = controller.force_feedback;
      config.hybrid.compensate_surface_friction = controller.friction_compensation;
      config.detect_collisions = controller.detect_collisions;
      Rig rig(SimConfig{}, kSeed, start.q, config);
      rig.sim().setTableHeightOffset(offset_m);
      rig.lockstep->enable(0.3);
      WipeTask::Params params;
      params.start_point = kSurfacePoint;
      params.rotation = home_rotation;
      params.posture = start.q;
      params.force_ramp_s = kForceRampS;
      const auto wipe_task = std::make_shared<WipeTask>(params);
      std::shared_ptr<const Task> task = wipe_task;
      if (controller.position_only) task = std::make_shared<PressIntoSurfaceTask>(wipe_task, 0.01, kForceRampS);

      Csv trace("wipe_trace_" + controller.name + "_" + std::to_string(std::lround(offset_m * 1000)) + "mm.csv",
                "task_time_s,force_true_n,force_ref_n,x,y,x_ref,y_ref");
      Reference ref;
      ref.q = start.q;
      double sum = 0, sum_sq_error = 0, minimum = INFINITY, maximum = 0, sum_sq_tangential = 0, samples = 0, lost = 0;
      long tick = 0;
      const double score_from_s = params.force_ramp_s + 0.5, score_until_s = params.force_ramp_s + params.laps * params.lap_period_s;
      const ControlStack& stack = *rig.stack;
      const bool started = rig.lockstep->execute(task, 30.0, [&](const RobotState&, const TorqueCommand&) {
        if (stack.supervisor().mode() != Mode::Executing) return;
        const double t = stack.taskTime();
        wipe_task->reference(t, ref);
        const double force = rig.sim().toolContactForce().z();
        const Vec3& x = stack.dynamics().tool_position;
        if (tick++ % 5 == 0) trace.row(t, force, ref.push_force_n, x.x(), x.y(), ref.tool_position.x(), ref.tool_position.y());
        if (t < score_from_s || t > score_until_s) return;
        sum += force;
        sum_sq_error += std::pow(force - params.push_force_n, 2);
        minimum = std::min(minimum, force);
        maximum = std::max(maximum, force);
        lost += force < kNoContactForceN ? 1.0 : 0.0;
        sum_sq_tangential += (x.head<2>() - ref.tool_position.head<2>()).squaredNorm();
        samples += 1;
      });
      if (!started || samples == 0) {
        std::printf("  %-26s offset %+3.0f mm: did not run\n%s", controller.name.c_str(), 1e3 * offset_m,
                    transitionLog(stack.supervisor()).c_str());
        continue;
      }
      int collision_trips = 0;
      for (std::size_t i = 0; i < stack.supervisor().retainedTransitions(); ++i)
        collision_trips += stack.supervisor().transition(i).why == Reason::Collision;
      csv.row(controller.name, 1e3 * offset_m, sum / samples, std::sqrt(sum_sq_error / samples), minimum, maximum,
              100.0 * lost / samples, 1e3 * std::sqrt(sum_sq_tangential / samples), collision_trips);
      std::printf("  %-26s offset %+3.0f mm  mean %5.2f N  rms err %5.2f N  min %5.2f  max %5.2f  lost %5.1f %%  tangential %.2f mm  trips %d\n",
                  controller.name.c_str(), 1e3 * offset_m, sum / samples, std::sqrt(sum_sq_error / samples), minimum, maximum,
                  100.0 * lost / samples, 1e3 * std::sqrt(sum_sq_tangential / samples), collision_trips);
    }
  }
}

const Register kWipe("wipe", wipe);

}  // namespace
}  // namespace g1rtc::experiments
