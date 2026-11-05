// ROS 2 front end for the G1 upper-body control stack.
//
// The 1 kHz loop does not run in an rclcpp callback. It runs on the Runtime's own threads, and
// this node only talks to it the way any non-real-time client must: goals go in through the
// TaskMailbox, operator commands through the supervisor's atomic requests, and telemetry comes
// out of a triple buffer that a wall timer polls. Planning runs on a worker thread so a slow RRT
// never blocks the executor.
//
//   topics   /joint_states (sensor_msgs/JointState)           publish_rate_hz
//            /g1rtc/tool_pose (geometry_msgs/PoseStamped)     tool frame in "world"
//            /g1rtc/residual (std_msgs/Float64MultiArray)     momentum-observer residual, N m
//            /g1rtc/mode (std_msgs/String)                    on change, transient local
//            /g1rtc/goal_pose (geometry_msgs/PoseStamped)     subscription: plan and move there.
//                A zero quaternion means "keep the home hand orientation".
//   services /g1rtc/enable, /g1rtc/stop, /g1rtc/reset (std_srvs/Trigger)
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "g1rtc/motion_planner.hpp"
#include "g1rtc/runtime.hpp"

using namespace std::chrono_literals;
using g1rtc::Mode;

class G1ArmRtcNode : public rclcpp::Node {
 public:
  G1ArmRtcNode()
      : rclcpp::Node("g1_arm_rtc"),
        sim_(std::make_unique<g1rtc::RobotSim>(G1RTC_MODEL_DIR "/g1_upper_body.xml", g1rtc::SimConfig{}, 1)),
        robot_(g1rtc::RobotDescription::fromMujoco(sim_->mjModel())),
        planner_(sim_->mjModel(), robot_) {
    const bool realtime = declare_parameter("realtime", true);
    const double publish_rate_hz = declare_parameter("publish_rate_hz", 100.0);
    sim_->reset(g1rtc::RobotDescription::homePosture());
    stack_ = std::make_unique<g1rtc::ControlStack>(robot_, g1rtc::StackConfig::defaultsG1());
    g1rtc::RuntimeConfig runtime_config;
    runtime_config.try_realtime = realtime;
    runtime_ = std::make_unique<g1rtc::Runtime>(*sim_, *stack_, runtime_config);

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/g1rtc/tool_pose", 10);
    residual_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/g1rtc/residual", 10);
    mode_pub_ = create_publisher<std_msgs::msg::String>("/g1rtc/mode", rclcpp::QoS(1).transient_local());
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/g1rtc/goal_pose", 10, [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr msg) { onGoal(*msg); });
    auto trigger = [this](const char* name, std::function<void()> request) {
      return create_service<std_srvs::srv::Trigger>(
          name, [this, request, name](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                                      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            request();
            // The request is handled on the next control tick; this reports the mode before it.
            response->success = true;
            response->message = std::string(name) + " requested in mode " + g1rtc::toString(stack_->supervisor().mode());
          });
    };
    enable_srv_ = trigger("/g1rtc/enable", [this] { stack_->supervisor().requestEnable(); });
    stop_srv_ = trigger("/g1rtc/stop", [this] { stack_->supervisor().requestStop(); });
    reset_srv_ = trigger("/g1rtc/reset", [this] { stack_->supervisor().requestReset(); });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / publish_rate_hz), [this] { publish(); });
    runtime_->start();
    RCLCPP_INFO(get_logger(), "control stack running (realtime requested: %s)", realtime ? "yes" : "no");
  }

  ~G1ArmRtcNode() override {
    if (planning_thread_.joinable()) planning_thread_.join();
    runtime_->stop();  // joins the RT threads before the stack and simulator are destroyed
  }

 private:
  void publish() {
    g1rtc::LogRecord record;
    if (!runtime_->pollTelemetry(record)) return;
    const int n = robot_.model.nv;
    {
      std::lock_guard<std::mutex> lock(measured_q_mutex_);  // the planning thread reads it
      measured_q_ = g1rtc::VecN(n);
      for (int i = 0; i < n; ++i) (*measured_q_)[i] = record.q[i];
    }
    const auto stamp = now();
    sensor_msgs::msg::JointState joints;
    joints.header.stamp = stamp;
    for (int i = 0; i < n; ++i) {
      joints.name.push_back(robot_.model.joint_names[static_cast<std::size_t>(i)]);
      joints.position.push_back(record.q[i]);
      joints.velocity.push_back(record.qd[i]);
      joints.effort.push_back(record.tau[i]);  // commanded, after saturation
    }
    joint_pub_->publish(joints);

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = "world";
    pose.pose.position.x = record.tool_position[0];
    pose.pose.position.y = record.tool_position[1];
    pose.pose.position.z = record.tool_position[2];
    pose.pose.orientation.w = record.tool_orientation_wxyz[0];
    pose.pose.orientation.x = record.tool_orientation_wxyz[1];
    pose.pose.orientation.y = record.tool_orientation_wxyz[2];
    pose.pose.orientation.z = record.tool_orientation_wxyz[3];
    pose_pub_->publish(pose);

    std_msgs::msg::Float64MultiArray residual;
    for (int i = 0; i < n; ++i) residual.data.push_back(record.residual[i]);
    residual_pub_->publish(residual);

    const auto mode = static_cast<Mode>(record.mode);
    if (!last_mode_ || *last_mode_ != mode) {
      std_msgs::msg::String text;
      text.data = g1rtc::toString(mode);
      mode_pub_->publish(text);
      last_mode_ = mode;
      RCLCPP_INFO(get_logger(), "mode: %s", text.data.c_str());
    }
  }

  void onGoal(const geometry_msgs::msg::PoseStamped& msg) {
    if (planning_.exchange(true)) {
      RCLCPP_WARN(get_logger(), "still planning the previous goal; ignoring this one");
      return;
    }
    if (planning_thread_.joinable()) planning_thread_.join();
    const g1rtc::Vec3 position(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
    const auto& o = msg.pose.orientation;
    const bool has_orientation = o.w * o.w + o.x * o.x + o.y * o.y + o.z * o.z > 0.5;  // unit, not zero
    const g1rtc::Mat3 rotation =
        has_orientation ? Eigen::Quaterniond(o.w, o.x, o.y, o.z).normalized().toRotationMatrix() : homeRotation();
    planning_thread_ = std::thread([this, position, rotation] {
      planAndExecute(position, rotation);
      planning_ = false;
    });
  }

  g1rtc::Mat3 homeRotation() const {
    g1rtc::rbd::DynamicsCache dyn(robot_.model);
    dyn.update(robot_.model, robot_.tool, g1rtc::RobotDescription::homePosture(), g1rtc::VecN::Zero(robot_.model.nv));
    return dyn.tool_rotation;
  }

  void planAndExecute(const g1rtc::Vec3& position, const g1rtc::Mat3& rotation) {
    std::optional<g1rtc::VecN> q_start;
    {
      std::lock_guard<std::mutex> lock(measured_q_mutex_);
      q_start = measured_q_;
    }
    if (!q_start) {
      RCLCPP_WARN(get_logger(), "no telemetry yet; ignoring the goal");
      return;
    }
    const g1rtc::PlannedMotion motion = planner_.planToPose(*q_start, position, rotation, ++plan_count_);
    if (!motion.ok()) {
      RCLCPP_WARN(get_logger(), "planning failed: %s", g1rtc::toString(motion.status));
      return;
    }
    RCLCPP_INFO(get_logger(), "planned in %.1f ms (%s), executing %.2f s trajectory", 1e3 * motion.planning_time_s,
                motion.used_rrt ? "RRT-Connect" : "straight", motion.trajectory.duration());
    stack_->mailbox().post(std::make_shared<g1rtc::JointTrajectoryTask>(motion.trajectory));
    std::this_thread::sleep_for(5ms);  // let the control thread see the task before the start request
    stack_->supervisor().requestStart();
    stack_->mailbox().collect();
  }

  std::unique_ptr<g1rtc::RobotSim> sim_;
  g1rtc::RobotDescription robot_;
  g1rtc::MotionPlanner planner_;  // declared after robot_, which it is built from
  std::unique_ptr<g1rtc::ControlStack> stack_;
  std::unique_ptr<g1rtc::Runtime> runtime_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr residual_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr enable_srv_, stop_srv_, reset_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::optional<Mode> last_mode_;
  std::mutex measured_q_mutex_;
  std::optional<g1rtc::VecN> measured_q_;
  std::atomic<bool> planning_{false};
  std::thread planning_thread_;
  std::uint64_t plan_count_ = 0;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<G1ArmRtcNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
