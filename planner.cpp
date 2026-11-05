#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "g1rtc/planning.hpp"

namespace g1rtc {

namespace {

struct Tree {
  std::vector<VecN> nodes;
  std::vector<int> parent;
  int nearest(const VecN& x) const {
    // Linear scan: the trees stay at a few hundred nodes here and validity checks dominate the
    // run time (RrtStats), so a k-d tree in 10-D would not pay for itself.
    int best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const double d = (nodes[i] - x).squaredNorm();
      if (d < best_distance) {
        best_distance = d;
        best = static_cast<int>(i);
      }
    }
    return best;
  }
  int add(const VecN& x, int parent_index) {
    nodes.push_back(x);
    parent.push_back(parent_index);
    return static_cast<int>(nodes.size()) - 1;
  }
  /// Nodes from `index` back to the root.
  std::vector<VecN> pathToRoot(int index) const {
    std::vector<VecN> out;
    for (int i = index; i >= 0; i = parent[static_cast<std::size_t>(i)]) out.push_back(nodes[static_cast<std::size_t>(i)]);
    return out;
  }
};

enum class Extension { Trapped, Advanced, Reached };

Extension extend(Tree& tree, const VecN& target, StateValidityChecker& checker, const RrtOptions& options, int& new_index) {
  const int nearest = tree.nearest(target);
  const VecN& from = tree.nodes[static_cast<std::size_t>(nearest)];
  const VecN direction = target - from;
  const double span = direction.cwiseAbs().maxCoeff();
  const bool reaches = span <= options.max_extension_rad;
  const VecN to = reaches ? target : VecN(from + direction * (options.max_extension_rad / span));
  if (!checker.segmentValid(from, to, options.edge_resolution_rad)) return Extension::Trapped;
  new_index = tree.add(to, nearest);
  return reaches ? Extension::Reached : Extension::Advanced;
}

}  // namespace

std::optional<std::vector<VecN>> rrtConnect(StateValidityChecker& checker, const VecN& start, const VecN& goal,
                                            const VecN& q_lower, const VecN& q_upper, const RrtOptions& options,
                                            RrtStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  const long checks0 = checker.checksPerformed();
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  Tree start_tree, goal_tree;
  start_tree.add(start, -1);
  goal_tree.add(goal, -1);
  Tree* growing = &start_tree;
  Tree* connecting = &goal_tree;
  std::optional<std::vector<VecN>> result;
  int it = 0;
  const bool endpoints_valid = checker.isValid(start) && checker.isValid(goal);
  for (; endpoints_valid && it < options.max_iterations; ++it) {
    VecN sample(start.size());
    for (int i = 0; i < sample.size(); ++i) sample[i] = q_lower[i] + unit(rng) * (q_upper[i] - q_lower[i]);
    int new_index = -1;
    if (extend(*growing, sample, checker, options, new_index) != Extension::Trapped) {
      const VecN target = growing->nodes[static_cast<std::size_t>(new_index)];
      int connect_index = -1;
      Extension status;
      do {
        status = extend(*connecting, target, checker, options, connect_index);
      } while (status == Extension::Advanced);
      if (status == Extension::Reached) {
        std::vector<VecN> path = growing->pathToRoot(new_index);
        std::reverse(path.begin(), path.end());
        const std::vector<VecN> rest = connecting->pathToRoot(connect_index);
        path.insert(path.end(), rest.begin() + 1, rest.end());  // rest[0] is the meeting node again
        if (growing != &start_tree) std::reverse(path.begin(), path.end());
        result = std::move(path);
        break;
      }
    }
    std::swap(growing, connecting);
  }
  if (stats) {
    stats->iterations = it;
    stats->nodes = static_cast<int>(start_tree.nodes.size() + goal_tree.nodes.size());
    stats->validity_checks = checker.checksPerformed() - checks0;
    stats->wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  return result;
}

double pathLength(const std::vector<VecN>& path) {
  double length = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) length += (path[i] - path[i - 1]).norm();
  return length;
}

namespace {
std::vector<double> cumulativeLength(const std::vector<VecN>& path) {
  std::vector<double> s(path.size(), 0.0);
  for (std::size_t i = 1; i < path.size(); ++i) s[i] = s[i - 1] + (path[i] - path[i - 1]).norm();
  return s;
}

/// Index of the segment [i-1, i] containing arc length s, clamped to a valid segment.
std::size_t segmentEndingAfter(const std::vector<double>& s_cumulative, double s) {
  const auto i = static_cast<std::size_t>(std::upper_bound(s_cumulative.begin(), s_cumulative.end(), s) - s_cumulative.begin());
  return std::clamp<std::size_t>(i, 1, s_cumulative.size() - 1);
}
}  // namespace

std::vector<VecN> shortcut(StateValidityChecker& checker, std::vector<VecN> path, int attempts, double resolution_rad,
                           std::mt19937_64& rng) {
  for (int k = 0; k < attempts && path.size() > 2; ++k) {
    const std::vector<double> s = cumulativeLength(path);
    std::uniform_real_distribution<double> along(0.0, s.back());
    double s1 = along(rng), s2 = along(rng);
    if (s1 > s2) std::swap(s1, s2);
    const std::size_t seg1 = segmentEndingAfter(s, s1), seg2 = segmentEndingAfter(s, s2);
    if (seg1 == seg2) continue;  // both points on one straight segment: nothing to gain
    auto pointAt = [&](double si, std::size_t seg) {
      const double h = s[seg] - s[seg - 1];
      const double fraction = h > 0 ? (si - s[seg - 1]) / h : 0.0;
      return VecN(path[seg - 1] + fraction * (path[seg] - path[seg - 1]));
    };
    const VecN p1 = pointAt(s1, seg1), p2 = pointAt(s2, seg2);
    if (!checker.segmentValid(p1, p2, resolution_rad)) continue;
    std::vector<VecN> shorter(path.begin(), path.begin() + static_cast<long>(seg1));
    shorter.push_back(p1);
    shorter.push_back(p2);
    shorter.insert(shorter.end(), path.begin() + static_cast<long>(seg2), path.end());
    path.swap(shorter);
  }
  return path;
}

// ------------------------------------------------------------------ natural cubic spline
JointSpline::JointSpline(const std::vector<VecN>& waypoints) : knot_q_(waypoints) {
  constexpr double kMinKnotSpacing = 1e-9;  // repeated waypoints would divide by zero
  const std::size_t n = waypoints.size();
  if (n < 2) throw std::invalid_argument("a spline needs at least two waypoints");
  knot_s_.assign(n, 0.0);
  for (std::size_t i = 1; i < n; ++i) knot_s_[i] = knot_s_[i - 1] + std::max((waypoints[i] - waypoints[i - 1]).norm(), kMinKnotSpacing);
  const auto dim = waypoints[0].size();
  knot_second_derivative_.assign(n, VecN::Zero(dim));
  if (n < 3) return;
  // Thomas algorithm for the knot second derivatives with natural ends (zero at both)
  std::vector<double> lower(n, 0.0), diagonal(n, 1.0), upper(n, 0.0);
  std::vector<VecN> rhs(n, VecN::Zero(dim));
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const double h_left = knot_s_[i] - knot_s_[i - 1], h_right = knot_s_[i + 1] - knot_s_[i];
    lower[i] = h_left / 6.0;
    diagonal[i] = (h_left + h_right) / 3.0;
    upper[i] = h_right / 6.0;
    rhs[i] = (waypoints[i + 1] - waypoints[i]) / h_right - (waypoints[i] - waypoints[i - 1]) / h_left;
  }
  for (std::size_t i = 1; i < n; ++i) {
    const double w = lower[i] / diagonal[i - 1];
    diagonal[i] -= w * upper[i - 1];
    rhs[i] -= w * rhs[i - 1];
  }
  knot_second_derivative_[n - 1] = rhs[n - 1] / diagonal[n - 1];
  for (std::size_t i = n - 1; i-- > 0;) knot_second_derivative_[i] = (rhs[i] - upper[i] * knot_second_derivative_[i + 1]) / diagonal[i];
}

std::size_t JointSpline::segment(double s) const {
  const auto it = std::upper_bound(knot_s_.begin(), knot_s_.end(), s);
  const long k = std::clamp<long>(it - knot_s_.begin() - 1, 0, static_cast<long>(knot_s_.size()) - 2);
  return static_cast<std::size_t>(k);
}

// Standard natural-spline segment formulas with A = s_{i+1} - s and B = s - s_i.
VecN JointSpline::configuration(double s) const {
  s = std::clamp(s, 0.0, length());
  const std::size_t i = segment(s);
  const double h = knot_s_[i + 1] - knot_s_[i], A = knot_s_[i + 1] - s, B = s - knot_s_[i];
  const VecN &m0 = knot_second_derivative_[i], &m1 = knot_second_derivative_[i + 1];
  return m0 * (A * A * A / (6 * h)) + m1 * (B * B * B / (6 * h)) + (knot_q_[i] / h - m0 * h / 6) * A +
         (knot_q_[i + 1] / h - m1 * h / 6) * B;
}

VecN JointSpline::tangent(double s) const {
  s = std::clamp(s, 0.0, length());
  const std::size_t i = segment(s);
  const double h = knot_s_[i + 1] - knot_s_[i], A = knot_s_[i + 1] - s, B = s - knot_s_[i];
  const VecN &m0 = knot_second_derivative_[i], &m1 = knot_second_derivative_[i + 1];
  return -m0 * (A * A / (2 * h)) + m1 * (B * B / (2 * h)) - (knot_q_[i] / h - m0 * h / 6) + (knot_q_[i + 1] / h - m1 * h / 6);
}

VecN JointSpline::curvature(double s) const {
  s = std::clamp(s, 0.0, length());
  const std::size_t i = segment(s);
  const double h = knot_s_[i + 1] - knot_s_[i], A = knot_s_[i + 1] - s, B = s - knot_s_[i];
  return knot_second_derivative_[i] * (A / h) + knot_second_derivative_[i + 1] * (B / h);
}

std::vector<VecN> dropCloseWaypoints(StateValidityChecker& checker, const std::vector<VecN>& waypoints, double min_spacing_rad,
                                     double resolution_rad) {
  if (waypoints.size() <= 2) return waypoints;
  std::vector<VecN> kept = {waypoints.front()};
  for (std::size_t i = 1; i + 1 < waypoints.size(); ++i) {
    const bool too_close = (waypoints[i] - kept.back()).norm() < min_spacing_rad;
    if (too_close && checker.segmentValid(kept.back(), waypoints[i + 1], resolution_rad)) continue;
    kept.push_back(waypoints[i]);
  }
  // the goal is never dropped; drop the waypoint before it instead if the two nearly coincide
  if (kept.size() > 1 && (waypoints.back() - kept.back()).norm() < min_spacing_rad &&
      checker.segmentValid(kept[kept.size() - 2], waypoints.back(), resolution_rad))
    kept.pop_back();
  kept.push_back(waypoints.back());
  return kept;
}

std::optional<JointSpline> validSpline(StateValidityChecker& checker, std::vector<VecN> waypoints, double resolution_rad,
                                       int max_refinements, double min_knot_spacing_rad) {
  waypoints = dropCloseWaypoints(checker, waypoints, min_knot_spacing_rad, resolution_rad);
  if (waypoints.size() == 2) waypoints.insert(waypoints.begin() + 1, VecN(0.5 * (waypoints[0] + waypoints[1])));
  for (int refinement = 0; refinement <= max_refinements; ++refinement) {
    JointSpline spline(waypoints);
    const int samples = std::max(2, static_cast<int>(std::ceil(spline.length() / resolution_rad)));
    std::optional<double> invalid_s;
    for (int k = 0; k <= samples && !invalid_s; ++k) {
      const double s = spline.length() * k / samples;
      if (!checker.isValid(spline.configuration(s))) invalid_s = s;
    }
    if (!invalid_s) return spline;
    const std::vector<double> s = cumulativeLength(waypoints);
    const std::size_t bad_segment = segmentEndingAfter(s, *invalid_s);
    std::vector<VecN> refined;
    for (std::size_t i = 0; i < waypoints.size(); ++i) {
      refined.push_back(waypoints[i]);
      const bool next_to_bad_segment = i + 2 >= bad_segment && i <= bad_segment;
      if (i + 1 < waypoints.size() && next_to_bad_segment) refined.push_back(VecN(0.5 * (waypoints[i] + waypoints[i + 1])));
    }
    waypoints.swap(refined);
  }
  return std::nullopt;
}

void Trajectory::sample(double t, VecN& q_out, VecN& qd_out, VecN& qdd_out) const {
  if (time_s.empty()) return;
  if (t <= time_s.front()) {
    q_out = q.front();
    qd_out = qd.front();
    qdd_out = qdd.front();
    return;
  }
  if (t >= time_s.back()) {
    q_out = q.back();
    qd_out = VecN::Zero(q.back().size());
    qdd_out = VecN::Zero(q.back().size());
    return;
  }
  const double period = time_s[1] - time_s[0];
  const auto i = std::min(static_cast<std::size_t>(t / period), time_s.size() - 2);
  const double a = (t - time_s[i]) / period;
  q_out = (1 - a) * q[i] + a * q[i + 1];
  qd_out = (1 - a) * qd[i] + a * qd[i + 1];
  qdd_out = (1 - a) * qdd[i] + a * qdd[i + 1];
}

}  // namespace g1rtc
