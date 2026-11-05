#!/usr/bin/env bash
# Start the node (no SCHED_FIFO inside an unprivileged container), run the checker, stop the node.
# The ROS setup scripts read unset variables, so -u is switched on only after sourcing them.
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
set -uo pipefail
ros2 launch g1_arm_rtc_ros g1_arm_rtc.launch.py realtime:=false > /tmp/node.log 2>&1 &
NODE=$!
sleep 4
ros2 run g1_arm_rtc_ros check_node.py
RC=$?
echo "---- node log (tail) ----"
tail -n 25 /tmp/node.log
# SIGTERM, not SIGINT: a background job in a non-interactive shell ignores SIGINT, and the launch
# process then never stops. Launch forwards a SIGINT to the node, which exits in about a second.
kill -TERM $NODE
for _ in $(seq 1 20); do kill -0 $NODE 2>/dev/null || break; sleep 0.5; done
kill -KILL $NODE 2>/dev/null
wait $NODE 2>/dev/null
exit $RC
