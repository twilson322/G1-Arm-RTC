// TOPP-RA: time-optimal path parameterization by reachability analysis (Pham & Pham, 2018).
//
// Along a geometric path q(s), with x = s_dot^2 and u = s_ddot, every limit used here is linear in
// (u, x) at a fixed s:
//   velocity      |q'_j| sqrt(x) <= v_j                  ->  x <= (v_j / |q'_j|)^2
//   acceleration  |q'_j u + q''_j x| <= a_j
//   torque        |A_j u + B_j x + C_j| <= tau_j,  A = M q',  B = M q'' + c(q, q'),  C = g(q)
// and between grid points x_{i+1} = x_i + 2 ds u_i. The backward pass computes, for every grid
// point, the interval of x from which the end (at rest) is still reachable, one 2-variable LP per
// point; the forward pass then takes the largest admissible u at each step.
#include <chrono>
#include <cmath>
#include <limits>

#include "g1rtc/planning.hpp"

namespace g1rtc {

namespace {

/// a * u + b * x <= c
struct LinearConstraint {
  double a, b, c;
};
using ConstraintSet = std::vector<LinearConstraint>;

/// Extreme value of x over {(u, x): all constraints}, by enumerating the polygon's vertices.
/// The polygons have under 100 edges and are bounded (x is boxed, u is pinned by the
/// reachability constraint), so this O(m^3) brute force costs about 0.1 ms per LP. Seidel's
/// O(m) algorithm would be the fix if planning time mattered.
std::optional<double> extremeX(const ConstraintSet& constraints, bool maximize) {
  constexpr double kParallel = 1e-12, kFeasibilityTolerance = 1e-9;
  std::optional<double> best;
  const std::size_t m = constraints.size();
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t l = k + 1; l < m; ++l) {
      const auto& p = constraints[k];
      const auto& q = constraints[l];
      const double det = p.a * q.b - q.a * p.b;
      if (std::abs(det) < kParallel) continue;
      const double u = (p.c * q.b - q.c * p.b) / det;
      const double x = (p.a * q.c - q.a * p.c) / det;
      bool feasible = true;
      for (const auto& r : constraints) {
        if (r.a * u + r.b * x > r.c + kFeasibilityTolerance * (1.0 + std::abs(r.c))) {
          feasible = false;
          break;
        }
      }
      if (feasible && (!best || (maximize ? x > *best : x < *best))) best = x;
    }
  }
  return best;
}

/// Velocity, acceleration and (optionally) torque limits at path parameter s.
ConstraintSet constraintsAt(const rbd::Model& m, const JointSpline& path, const ToppLimits& limits, double s,
                            rbd::Kinematics<double>& kin, rbd::Workspace<double>& work) {
  constexpr double kNegligibleTangent = 1e-9;
  constexpr double kUnboundedSpeedSquared = 1e6;
  const VecN q = path.configuration(s), dq = path.tangent(s), d2q = path.curvature(s);
  ConstraintSet set;
  double max_speed_squared = kUnboundedSpeedSquared;
  for (int j = 0; j < m.nv; ++j) {
    if (std::abs(dq[j]) > kNegligibleTangent) max_speed_squared = std::min(max_speed_squared, std::pow(limits.qd_max_rad_s[j] / std::abs(dq[j]), 2));
    set.push_back({dq[j], d2q[j], limits.qdd_max_rad_s2[j]});
    set.push_back({-dq[j], -d2q[j], limits.qdd_max_rad_s2[j]});
  }
  if (limits.enforce_torque) {
    rbd::forwardKinematics(m, q, kin);
    const VecN zero = VecN::Zero(m.nv);
    VecN torque_per_u, torque_per_x, gravity;
    rbd::rnea(m, kin, zero, dq, false, work, torque_per_u);  // M q'
    rbd::rnea(m, kin, dq, d2q, false, work, torque_per_x);   // M q'' + c(q, q')
    rbd::rnea(m, kin, zero, zero, true, work, gravity);
    for (int j = 0; j < m.nv; ++j) {
      set.push_back({torque_per_u[j], torque_per_x[j], limits.tau_max_nm[j] - gravity[j]});
      set.push_back({-torque_per_u[j], -torque_per_x[j], limits.tau_max_nm[j] + gravity[j]});
    }
  }
  set.push_back({0.0, 1.0, max_speed_squared});
  set.push_back({0.0, -1.0, 0.0});
  return set;
}

/// Sample s(t) from the grid solution at a fixed period, with q, qd, qdd from the chain rule.
Trajectory resample(const JointSpline& path, const std::vector<double>& x, const std::vector<double>& u, double ds,
                    double period_s) {
  const std::size_t grid = u.size();
  std::vector<double> grid_time(grid + 1, 0.0);
  for (std::size_t i = 0; i < grid; ++i) {
    const double speed_sum = std::sqrt(x[i]) + std::sqrt(x[i + 1]);
    grid_time[i + 1] = grid_time[i] + (speed_sum > 1e-12 ? 2 * ds / speed_sum : 0.0);
  }
  Trajectory out;
  std::size_t seg = 0;
  const double end = grid_time.back();
  for (double t = 0.0; t <= end + 1e-12; t += period_s) {
    while (seg + 1 < grid && grid_time[seg + 1] <= t) ++seg;
    const double dt = t - grid_time[seg];
    const double s_dot0 = std::sqrt(x[seg]);
    const double s = std::clamp(ds * static_cast<double>(seg) + s_dot0 * dt + 0.5 * u[seg] * dt * dt, 0.0, path.length());
    const double s_dot = std::max(s_dot0 + u[seg] * dt, 0.0);
    const VecN tangent = path.tangent(s);
    out.time_s.push_back(t);
    out.q.push_back(path.configuration(s));
    out.qd.push_back(tangent * s_dot);
    out.qdd.push_back(tangent * u[seg] + path.curvature(s) * s_dot * s_dot);
  }
  const auto n = out.q.back().size();
  out.time_s.push_back(out.time_s.back() + period_s);
  out.q.push_back(path.configuration(path.length()));
  out.qd.push_back(VecN::Zero(n));
  out.qdd.push_back(VecN::Zero(n));
  return out;
}

}  // namespace

std::optional<Trajectory> toppra(const rbd::Model& m, const JointSpline& path, const ToppLimits& limits, int grid_points,
                                 double sample_period_s, ToppStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto finish = [&](ToppStatus status) {
    if (stats) {
      stats->status = status;
      stats->wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
  };
  const auto N = static_cast<std::size_t>(grid_points);
  const double ds = path.length() / grid_points;
  rbd::Kinematics<double> kin(m);
  rbd::Workspace<double> work(m);
  std::vector<ConstraintSet> at_point(N + 1);
  for (std::size_t i = 0; i <= N; ++i) at_point[i] = constraintsAt(m, path, limits, ds * static_cast<double>(i), kin, work);

  // Segment i runs at constant u_i from x_i to x_i + 2 ds u_i, so (interpolation scheme) the
  // limits at point i+1 also apply to (u_i, x_i): substitute x_{i+1} = x_i + 2 ds u_i.
  std::vector<ConstraintSet> on_segment(N);
  for (std::size_t i = 0; i < N; ++i) {
    on_segment[i] = at_point[i];
    if (limits.interpolation_constraints)
      for (const auto& c : at_point[i + 1]) on_segment[i].push_back({c.a + 2 * ds * c.b, c.b, c.c});
  }

  // backward pass: controllable sets [x_min_i, x_max_i] that can still stop at the end
  std::vector<double> x_min(N + 1, 0.0), x_max(N + 1, 0.0);
  for (std::size_t i = N; i-- > 0;) {
    ConstraintSet set = on_segment[i];
    set.push_back({2 * ds, 1.0, x_max[i + 1]});
    set.push_back({-2 * ds, -1.0, -x_min[i + 1]});
    const auto lo = extremeX(set, false), hi = extremeX(set, true);
    if (!lo || !hi) {
      finish(ToppStatus::InfeasibleAlongPath);
      return std::nullopt;
    }
    x_min[i] = std::max(0.0, *lo);
    x_max[i] = *hi;
  }
  if (x_min[0] > 1e-9) {
    finish(ToppStatus::CannotStartFromRest);
    return std::nullopt;
  }

  // forward pass: largest admissible path acceleration that stays controllable
  std::vector<double> x(N + 1, 0.0), u(N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    double u_low = -std::numeric_limits<double>::infinity(), u_high = std::numeric_limits<double>::infinity();
    for (const auto& c : on_segment[i]) {
      const double slack = c.c - c.b * x[i];
      if (c.a > 1e-12) u_high = std::min(u_high, slack / c.a);
      else if (c.a < -1e-12) u_low = std::max(u_low, slack / c.a);
    }
    u_high = std::min(u_high, (x_max[i + 1] - x[i]) / (2 * ds));
    u_low = std::max(u_low, (x_min[i + 1] - x[i]) / (2 * ds));
    // u_low > u_high only through round-off; the clamp below keeps x inside the controllable set
    const double u_chosen = std::max(u_high, u_low);
    x[i + 1] = std::clamp(x[i] + 2 * ds * u_chosen, x_min[i + 1], x_max[i + 1]);
    u[i] = (x[i + 1] - x[i]) / (2 * ds);
  }
  Trajectory trajectory = resample(path, x, u, ds, sample_period_s);
  finish(ToppStatus::Ok);
  return trajectory;
}

double peakTorqueUtilization(const rbd::Model& m, const Trajectory& trajectory, const VecN& tau_limit_nm) {
  rbd::Kinematics<double> kin(m);
  rbd::Workspace<double> work(m);
  VecN tau(m.nv);
  double peak = 0.0;
  for (std::size_t i = 0; i < trajectory.time_s.size(); ++i) {
    rbd::forwardKinematics(m, trajectory.q[i], kin);
    rbd::rnea(m, kin, trajectory.qd[i], trajectory.qdd[i], true, work, tau);
    peak = std::max(peak, tau.cwiseAbs().cwiseQuotient(tau_limit_nm).maxCoeff());
  }
  return peak;
}

Trajectory uniformTrapezoid(const JointSpline& path, const ToppLimits& limits, double sample_period_s) {
  constexpr int kScanSamples = 400;
  const int n = static_cast<int>(limits.qd_max_rad_s.size());
  VecN max_tangent = VecN::Zero(n), max_curvature = VecN::Zero(n);
  for (int k = 0; k <= kScanSamples; ++k) {
    const double s = path.length() * k / kScanSamples;
    max_tangent = max_tangent.cwiseMax(path.tangent(s).cwiseAbs());
    max_curvature = max_curvature.cwiseMax(path.curvature(s).cwiseAbs());
  }
  // Peak s_dot from the velocity limits, and from half of each acceleration limit for the
  // centripetal term q'' s_dot^2; the other half is left for s_ddot.
  double peak_speed = std::numeric_limits<double>::infinity(), peak_acceleration = std::numeric_limits<double>::infinity();
  for (int j = 0; j < n; ++j) {
    if (max_tangent[j] > 1e-9) peak_speed = std::min(peak_speed, limits.qd_max_rad_s[j] / max_tangent[j]);
    if (max_curvature[j] > 1e-9) peak_speed = std::min(peak_speed, std::sqrt(0.5 * limits.qdd_max_rad_s2[j] / max_curvature[j]));
  }
  for (int j = 0; j < n; ++j)
    if (max_tangent[j] > 1e-9)
      peak_acceleration = std::min(peak_acceleration, (limits.qdd_max_rad_s2[j] - max_curvature[j] * peak_speed * peak_speed) / max_tangent[j]);
  const double L = path.length();
  double ramp_time = peak_speed / peak_acceleration;
  if (peak_acceleration * ramp_time * ramp_time > L) {  // too short to reach peak speed: triangle
    ramp_time = std::sqrt(L / peak_acceleration);
    peak_speed = peak_acceleration * ramp_time;
  }
  const double cruise_time = (L - peak_speed * ramp_time) / peak_speed;
  const double total = 2 * ramp_time + cruise_time;
  Trajectory out;
  for (double t = 0.0; t <= total + 1e-12; t += sample_period_s) {
    double s, s_dot, s_ddot;
    if (t < ramp_time) {
      s = 0.5 * peak_acceleration * t * t;
      s_dot = peak_acceleration * t;
      s_ddot = peak_acceleration;
    } else if (t < ramp_time + cruise_time) {
      s = 0.5 * peak_speed * ramp_time + peak_speed * (t - ramp_time);
      s_dot = peak_speed;
      s_ddot = 0.0;
    } else {
      const double remaining = total - t;
      s = L - 0.5 * peak_acceleration * remaining * remaining;
      s_dot = peak_acceleration * remaining;
      s_ddot = -peak_acceleration;
    }
    const VecN tangent = path.tangent(s);
    out.time_s.push_back(t);
    out.q.push_back(path.configuration(s));
    out.qd.push_back(tangent * s_dot);
    out.qdd.push_back(tangent * s_ddot + path.curvature(s) * s_dot * s_dot);
  }
  out.time_s.push_back(out.time_s.back() + sample_period_s);
  out.q.push_back(path.configuration(L));
  out.qd.push_back(VecN::Zero(n));
  out.qdd.push_back(VecN::Zero(n));
  return out;
}

}  // namespace g1rtc
