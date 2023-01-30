#pragma once

#include <string>

namespace mononn_engine {
namespace tuning {
namespace profiler {
struct ProfilingResult {
  float time_in_us = -1;
  std::string output;

  bool codegen_success;
  bool build_success;
  bool profile_success;

  std::string profiling_directory;

  void verify() const;
};
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine