// Motion generation, off the real-time path: IK, validity checking, RRT-Connect, shortcutting, a
// C2 spline through the waypoints and time-optimal parameterization (TOPP-RA) under joint
// velocity, acceleration and torque limits. All of it runs in a planning thread; only the
// resulting trajectory samples cross into the 1 kHz loop.
#pragma once

#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "g1rtc/rbd.hpp"

struct mjModel_;
struct mjData_;

namespace g1rtc {

// ---------------------------------------------------------------- inverse kinematics
struct IkOptions {
  int max_iterations = 300;
  double position_tolerance_m = 1e-4;
  double orientation_tolerance_rad = 1e-3;
  double max_step_rad = 0.2;              // per iteration, inf-norm
  double posture_gain = 0.1;              // pull toward q_rest in the null space, per iteration
  double joint_limit_margin_rad = 0.02;
  std::vector<int> locked_joints;
};

struct IkResult {
  VecN q;
  bool converged = false;
  int iterations = 0;
  double position_error_m = 0.0, orientation_error_rad = 0.0;
};

/// Damped least squares on the tool pose, with a null-space pull toward q_rest and joint-limit
/// clamping. The damping grows only near singularities.
IkResult solveIk(const rbd::Model& m, const rbd::Frame& tool, const Vec3& target_position, const Mat3& target_rotation,
                 const VecN& q_initial, const VecN& q_rest, const IkOptions& options = {});

// ---------------------------------------------------------------- validity checking
/// Robot body pairs whose collision is ignored. Inflating every geom by the planning clearance
/// makes some neighbouring links "touch" in every configuration (the elbow and wrist-pitch shells,
/// the two wrist links). Like MoveIt's setup assistant, `sample` disables robot-robot pairs that
/// are within the clearance at the zero pose or in at least 95 % of random configurations.
/// Pairs involving the environment are never disabled. Robot bodies are those in the kinematic
/// tree that carries the joints; every other body is environment.
class AllowedCollisionMatrix {
 public:
  static AllowedCollisionMatrix sample(const mjModel_* m, double clearance_m, int samples = 300, std::uint64_t seed = 7);
  bool allowed(int body_a, int body_b) const;
  int allowedPairCount() const { return allowed_pairs_; }

 private:
  int nbody_ = 0, allowed_pairs_ = 0;
  std::vector<char> allowed_;  // nbody x nbody, upper triangle used
};

enum class Validity : std::uint8_t { Valid, NearJointLimit, InCollision };
const char* toString(Validity v);

/// A configuration is valid if it is inside the joint limits (minus a margin) and every geom is
/// at least `clearance_m` from everything it may collide with. Owns a private copy of the MuJoCo
/// model, so one instance per thread; constructing one is cheap once the ACM exists.
class StateValidityChecker {
 public:
  StateValidityChecker(const mjModel_* m, double clearance_m, AllowedCollisionMatrix acm,
                       double joint_limit_margin_rad = 0.02);
  ~StateValidityChecker();
  StateValidityChecker(const StateValidityChecker&) = delete;
  StateValidityChecker& operator=(const StateValidityChecker&) = delete;
  Validity classify(const VecN& q);
  bool isValid(const VecN& q) { return classify(q) == Validity::Valid; }
  /// Checks the straight joint-space segment at `resolution_rad` (inf-norm) spacing, coarse to
  /// fine, so a collision is usually found after a few checks.
  bool segmentValid(const VecN& from, const VecN& to, double resolution_rad);
  long checksPerformed() const { return checks_; }
  const AllowedCollisionMatrix& allowedCollisions() const { return acm_; }

 private:
  mjModel_* m_;
  mjData_* d_;
  AllowedCollisionMatrix acm_;
  double joint_limit_margin_rad_;
  long checks_ = 0;
};

// ---------------------------------------------------------------- RRT-Connect
struct RrtOptions {
  double max_extension_rad = 0.25;  // inf-norm
  double edge_resolution_rad = 0.02;
  int max_iterations = 20000;
  std::uint64_t seed = 1;
};
struct RrtStats {
  int iterations = 0;
  int nodes = 0;
  long validity_checks = 0;
  double wall_time_s = 0.0;
};
/// Bidirectional RRT with greedy connection (Kuffner & LaValle 2000), sampling uniformly in
/// [q_lower, q_upper]. Returns nullopt if either endpoint is invalid or no path is found.
std::optional<std::vector<VecN>> rrtConnect(StateValidityChecker& checker, const VecN& start, const VecN& goal,
                                            const VecN& q_lower, const VecN& q_upper, const RrtOptions& options,
                                            RrtStats* stats = nullptr);
/// Random shortcutting between points sampled along the path by arc length.
std::vector<VecN> shortcut(StateValidityChecker& checker, std::vector<VecN> path, int attempts, double resolution_rad,
                           std::mt19937_64& rng);
double pathLength(const std::vector<VecN>& path);

// ---------------------------------------------------------------- geometric path
/// Natural cubic spline through waypoints, parameterized by cumulative chord length s (rad).
class JointSpline {
 public:
  explicit JointSpline(const std::vector<VecN>& waypoints);
  double length() const { return knot_s_.back(); }
  VecN configuration(double s) const;
  VecN tangent(double s) const;    // dq/ds
  VecN curvature(double s) const;  // d2q/ds2

 private:
  std::size_t segment(double s) const;
  std::vector<double> knot_s_;
  std::vector<VecN> knot_q_, knot_second_derivative_;
};

/// Drop interior waypoints closer than `min_spacing_rad` to the previous kept one, where the
/// straight segment that skips them is valid. Shortcutting leaves such near-duplicates, and a
/// cubic spline through two knots a few mrad apart bends sharply there: on the shelf path one
/// knot 3 mrad from the start gave |q''| = 690 and a torque peak TOPP-RA's grid stepped over.
std::vector<VecN> dropCloseWaypoints(StateValidityChecker& checker, const std::vector<VecN>& waypoints, double min_spacing_rad,
                                     double resolution_rad);

/// Fit a spline through the waypoints (after dropCloseWaypoints) and check it densely. Where it
/// cuts a corner into an obstacle or past a joint limit, add the midpoints of the neighbouring
/// straight segments as knots, which pulls it toward the valid polyline, and refit.
std::optional<JointSpline> validSpline(StateValidityChecker& checker, std::vector<VecN> waypoints, double resolution_rad,
                                       int max_refinements = 6, double min_knot_spacing_rad = 0.05);

// ---------------------------------------------------------------- time parameterization
struct ToppLimits {
  VecN qd_max_rad_s, qdd_max_rad_s2;
  VecN tau_max_nm;  // used only if enforce_torque
  bool enforce_torque = false;
  /// Impose each grid point's limits on the segment before it too (first-order interpolation in
  /// Pham & Pham). Collocation alone overshot the torque limit 4x between points on coarse grids.
  bool interpolation_constraints = true;
};

struct Trajectory {
  std::vector<double> time_s;
  std::vector<VecN> q, qd, qdd;
  double duration() const { return time_s.empty() ? 0.0 : time_s.back(); }
  /// Linear interpolation between the stored samples; holds the final pose at rest afterwards.
  void sample(double t, VecN& q_out, VecN& qd_out, VecN& qdd_out) const;
};

enum class ToppStatus : std::uint8_t { Ok, InfeasibleAlongPath, CannotStartFromRest };
struct ToppStats {
  ToppStatus status = ToppStatus::Ok;
  double wall_time_s = 0.0;
};

/// TOPP-RA (Pham & Pham 2018). With x = s_dot^2 and u = s_ddot every limit is linear in (u, x),
/// including torque: tau = M q' u + (M q'' + c(q, q')) x + g. The backward pass computes the
/// controllable set of x at each grid point with a 2-variable LP, the forward pass takes the
/// largest admissible u. The path starts and ends at rest.
std::optional<Trajectory> toppra(const rbd::Model& m, const JointSpline& path, const ToppLimits& limits, int grid_points,
                                 double sample_period_s, ToppStats* stats = nullptr);

/// Largest |tau_j| / tau_limit_j over the trajectory's samples, with tau from inverse dynamics on
/// `m`. TOPP-RA enforces limits at its grid points only; this checks what happens in between.
double peakTorqueUtilization(const rbd::Model& m, const Trajectory& trajectory, const VecN& tau_limit_nm);

/// Baseline: the same path with one trapezoidal s(t) profile, its peak speed and acceleration set
/// by the worst-case path derivatives anywhere on the path (what a simple implementation does).
Trajectory uniformTrapezoid(const JointSpline& path, const ToppLimits& limits, double sample_period_s);

}  // namespace g1rtc
