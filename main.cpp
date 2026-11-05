// Deterministic experiments. Usage: g1rtc_experiments [name|all] [output_dir]
#include <mujoco/mujoco.h>

#include <cstdio>
#include <exception>

#include "support.hpp"

int main(int argc, char** argv) {
  using namespace g1rtc::experiments;
  const std::string which = argc > 1 ? argv[1] : "all";
  if (argc > 2) outputDirectory() = argv[2];
  if (which != "all" && !registry().count(which)) {
    std::fprintf(stderr, "unknown experiment '%s'; available: all", which.c_str());
    for (const auto& [name, unused] : registry()) std::fprintf(stderr, " %s", name.c_str());
    std::fprintf(stderr, "\n");
    return 2;
  }
  std::filesystem::create_directories(outputDirectory());
  mju_user_warning = [](const char* message) { std::fprintf(stderr, "mujoco warning: %s\n", message); };
  try {
    for (const auto& [name, experiment] : registry())
      if (which == "all" || which == name) experiment();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "experiment failed: %s\n", e.what());
    return 1;
  }
  return 0;
}
