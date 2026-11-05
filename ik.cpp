#include <algorithm>

#include "g1rtc/planning.hpp"

namespace g1rtc {

IkResult solveIk(const rbd::Model& m, const rbd::Frame& tool, const Vec3& target_position, const Mat3& target_rotation,
                 const VecN& q_initial, const VecN& q_rest, const IkOptions& options) {
  // Damping is 1e-4 away from singularities and ramps to 1e-2 as det(J J^T) falls below 1e-3,
  // a smooth version of Nakamura's singularity-robust inverse.
  constexpr double kBaseDamping = 1e-4, kSingularDamping = 1e-2, kManipulabilityThreshold = 1e-3;
  rbd::Kinematics<double> kin(m);
  Mat6N J;
  IkResult result;
  VecN q = q_initial;
  for (int it = 0; it < options.max_iterations; ++it) {
    rbd::forwardKinematics(m, q, kin);
    Vec3 position;
    Mat3 rotation;
    rbd::framePose(kin, tool, position, rotation);
    Vec6 error;
    error << target_position - position, rotationError(target_rotation, rotation);
    result.iterations = it;
    result.position_error_m = error.head<3>().norm();
    result.orientation_error_rad = error.tail<3>().norm();
    if (result.position_error_m < options.position_tolerance_m && result.orientation_error_rad < options.orientation_tolerance_rad) {
      result.converged = true;
      break;
    }
    rbd::pointJacobian(m, kin, tool.body, position, J);
    for (int j : options.locked_joints) J.col(j).setZero();
    const Mat6 JJt = J * J.transpose();
    const double damping = kBaseDamping + kSingularDamping * std::max(0.0, kManipulabilityThreshold - JJt.determinant()) /
                                              kManipulabilityThreshold;
    const MatN6 J_pinv = J.transpose() * (JJt + damping * Mat6::Identity()).ldlt().solve(Mat6::Identity());
    VecN step = J_pinv * error;
    VecN posture_pull = options.posture_gain * (q_rest - q);
    for (int j : options.locked_joints) posture_pull[j] = 0.0;
    step += posture_pull - J_pinv * (J * posture_pull);
    const double largest = step.cwiseAbs().maxCoeff();
    if (largest > options.max_step_rad) step *= options.max_step_rad / largest;
    q += step;
    for (int i = 0; i < m.nv; ++i)
      q[i] = std::clamp(q[i], m.q_min[i] + options.joint_limit_margin_rad, m.q_max[i] - options.joint_limit_margin_rad);
  }
  result.q = q;
  return result;
}

}  // namespace g1rtc
