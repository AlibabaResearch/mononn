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

#include <future>
#include <mutex>

#include "mononn_engine/tuning/profiler/profiling_result.h"
#include "mononn_engine/tuning/profiler/thread_pool.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace tuning {
namespace profiler {
class ParallelProfilingQueue {
 public:
  struct NCUResult {
    float time_in_us;
  };

  ParallelProfilingQueue(int n_threads) : thread_pool(n_threads) {}

  using GraphSpecification =
      tensorflow::mononn_extra::proto::GraphSpecification;
  // Non blocking post
  std::future<ProfilingResult> post(
      GraphSpecification const* graph_spec,
      std::vector<std::string> host_codegen_disabled_pass = {},
      std::vector<std::string> optimization_disabled_pass = {});

 private:
  ThreadPool thread_pool;

  std::mutex thread_mutex;
  std::mutex profile_mutex[8];

  NCUResult parse_ncu_result(std::string str) const;
  NCUResult parse_console_result(std::string str) const;
};
}  // namespace profiler
}  // namespace tuning
}  // namespace mononn_engine
