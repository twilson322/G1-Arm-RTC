// What the controller knows about the robot: its dynamics model and the frames it uses.
//
// Frames (all right-handed, world z up, gravity -z):
//   world      fixed; the pelvis is welded 0.793 m above it.
//   tool       "tool_tip" site, the centre of the 3 cm sponge on the right hand. Cartesian
//              references and the tool Jacobian are for this frame.
//   ft_sensor  "ft_site", where the wrist F/T sensor measures; it coincides with the tool frame
//              in this scene, but the code does not rely on that.
// Units are SI throughout (m, rad, s, kg, N, N m) unless a name says otherwise.
#pragma once

#include "g1rtc/rbd.hpp"

struct mjModel_;

namespace g1rtc {

struct RobotDescription {
  rbd::Model model;
  rbd::Frame tool;
  rbd::Frame ft_sensor;
  static RobotDescription fromMujoco(const mjModel_* m);
  /// The posture every experiment starts from and the default null-space posture.
  static VecN homePosture();
};

}  // namespace g1rtc
