#!/usr/bin/env python3
"""End-to-end check of the ROS 2 interface against a running g1_arm_rtc_node.

enable -> joint_states rate -> goal under the shelf -> goal above it (needs RRT around the shelf)
-> stop service -> reset service. Exits non-zero on the first failed expectation.
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Checker(Node):
    def __init__(self):
        super().__init__("g1rtc_checker")
        self.mode = None
        self.modes = []
        self.pose = None
        self.js_count = 0
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/g1rtc/mode", self.on_mode, qos)
        self.create_subscription(PoseStamped, "/g1rtc/tool_pose", self.on_pose, 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/g1rtc/goal_pose", 10)

    def on_mode(self, m):
        self.mode = m.data
        self.modes.append(m.data)

    def on_pose(self, p):
        self.pose = (p.pose.position.x, p.pose.position.y, p.pose.position.z)

    def on_js(self, _):
        self.js_count += 1

    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_mode(self, want, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.mode == want:
                return True
        return False

    def call(self, name):
        cli = self.create_client(Trigger, name)
        if not cli.wait_for_service(timeout_sec=10.0):
            return None
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result()

    def goal(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
        msg.pose.orientation.w = 0.0  # zero quaternion: keep the home hand orientation
        self.goal_pub.publish(msg)


def main():
    rclpy.init()
    n = Checker()
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    check("node reports a mode", n.wait_mode("disabled", 20.0) or n.mode is not None, f"(mode={n.mode})")
    r = n.call("/g1rtc/enable")
    check("enable service", r is not None and r.success, r.message if r else "")
    check("reaches hold", n.wait_mode("hold", 5.0))
    n.js_count = 0
    n.spin_for(2.0)
    rate = n.js_count / 2.0
    check("joint_states rate", 80.0 <= rate <= 120.0, f"({rate:.1f} Hz)")
    for name, g in [("under the shelf", (0.30, -0.28, 0.90)), ("above the shelf", (0.30, -0.28, 1.10))]:
        n.goal(*g)
        started = n.wait_mode("executing", 10.0)
        done = started and n.wait_mode("hold", 10.0)
        n.spin_for(0.3)
        err = math.dist(n.pose, g) if n.pose else float("inf")
        check(f"move {name}", done and err < 0.01, f"(final error {1000 * err:.1f} mm)")
    r = n.call("/g1rtc/stop")
    check("stop -> safe_stop", r is not None and n.wait_mode("safe_stop", 3.0))
    r = n.call("/g1rtc/reset")
    check("reset -> hold", r is not None and n.wait_mode("hold", 3.0))
    print("mode sequence:", " -> ".join(n.modes), flush=True)
    rclpy.shutdown()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
