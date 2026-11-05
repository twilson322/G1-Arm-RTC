cmake_minimum_required(VERSION 3.22)
project(g1_arm_rtc_ros LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# The control stack itself lives two directories up and is built here without its tests.
set(G1RTC_TESTS OFF CACHE BOOL "" FORCE)
set(G1RTC_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. g1rtc_core EXCLUDE_FROM_ALL)

add_executable(g1_arm_rtc_node src/g1_arm_rtc_node.cpp)
target_link_libraries(g1_arm_rtc_node g1rtc)
ament_target_dependencies(g1_arm_rtc_node rclcpp sensor_msgs geometry_msgs std_msgs std_srvs)
# libmujoco lives in the pip wheel's directory, outside the install tree. Keep that directory in
# the installed binary's RPATH, or the node fails to start with "libmujoco.so not found".
set_target_properties(g1_arm_rtc_node PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS g1_arm_rtc_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/check_node.py DESTINATION lib/${PROJECT_NAME})
ament_package()
