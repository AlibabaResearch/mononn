// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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