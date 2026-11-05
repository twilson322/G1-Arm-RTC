// Reaction strategies when the figure-8 runs into an obstacle the planner did not know about.
#include <cmath>
#include <cstdio>
#include <numbers>

#include "support.hpp"

namespace g1rtc::experiments {
namespace {

constexpr std::uint64_t kSeed = 21;
constexpr double kContactForceThresholdN = 0.5;
constexpr double kScoringWindowS = 1.0;  // force statistics over the first second after contact

void reaction() {
  std::printf("== collision reaction with an unplanned obstacle\n");
  Csv csv("reaction.csv", "reaction,detected,latency_ms,peak_force_n,impulse_1s_ns,contact_time_ms,final_mode");
  for (ReflexKind reflex : {ReflexKind::Ignore, ReflexKind::Stop, ReflexKind::Float, ReflexKind::Retract}) {
    StackConfig config = StackConfig::defaultsG1();
    config.supervisor.reflex = reflex;
    config.supervisor.reflex_duration_s = 1.0;
    Rig rig(SimConfig{}, kSeed, RobotDescription::homePosture(), config);
    rig.lockstep->enable(0.5);
    const Figure8Task::Params params = figure8At(rig.world.robot, rig.sim().trueQ());
    // A forearm-sized capsule (radius 4.5 cm) lying along x across the left lobe of the figure, so
    // the sponge's path passes 5 cm inside it: clear of the arm at the start, hit about 0.6 s in.
    const Mat3 lying_along_x = Eigen::AngleAxisd(std::numbers::pi / 2, Vec3::UnitY()).toRotationMatrix();
    rig.sim().setIntruderPose(params.center + Vec3(0.0, 0.12, 0.0), lying_along_x);
    const auto task = std::make_shared<Figure8Task>(params);

    Csv trace(std::string("reaction_trace_") + toString(reflex) + ".csv", "time_s,contact_force_n,mode,max_abs_residual_nm");
    double contact_s = -1, detected_s = -1, peak_n = 0, impulse_ns = 0, contact_ms = 0;
    long tick = 0;
    ControlStack& stack = *rig.stack;
    stack.mailbox().post(task);
    rig.lockstep->run(2 * Lockstep::kControlPeriodS);
    stack.supervisor().requestStart();
    // Run for the whole task (plus a margin so it can finish) even after the reflex, to see where
    // each strategy ends up.
    constexpr double kAfterTaskS = 0.1;
    rig.lockstep->run(task->duration() + kAfterTaskS, [&](const RobotState& state, const TorqueCommand&) {
      const double force = rig.sim().contactForceWith("intruder_geom");
      if (force > kContactForceThresholdN && contact_s < 0) contact_s = state.time_s;
      if (stack.collisionDetectedThisTick() && detected_s < 0) detected_s = state.time_s;
      if (contact_s >= 0 && state.time_s - contact_s <= kScoringWindowS) {
        peak_n = std::max(peak_n, force);
        impulse_ns += force * Lockstep::kControlPeriodS;
        contact_ms += force > kContactForceThresholdN ? 1e3 * Lockstep::kControlPeriodS : 0.0;
      }
      if (tick++ % 2 == 0)
        trace.row(state.time_s, force, static_cast<int>(stack.supervisor().mode()), stack.residual().cwiseAbs().maxCoeff());
    });
    const double latency_ms = detected_s >= 0 && contact_s >= 0 ? 1e3 * (detected_s - contact_s) : NAN;
    const char* final_mode = toString(stack.supervisor().mode());
    csv.row(toString(reflex), detected_s >= 0, latency_ms, peak_n, impulse_ns, contact_ms, final_mode);
    std::printf("  %-8s contact at %.3f s  detected %d (%.0f ms)  peak %.1f N  impulse (1 s) %.2f N s  contact %.0f ms  final %s\n",
                toString(reflex), contact_s, detected_s >= 0, latency_ms, peak_n, impulse_ns, contact_ms, final_mode);
  }
}

const Register kReaction("reaction", reaction);

}  // namespace
}  // namespace g1rtc::experiments
