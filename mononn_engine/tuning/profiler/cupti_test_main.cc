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

#include <cstdio>

#include "mononn_engine/tuning/profiler/cupti_profiling_session.h"
using CuptiProfilingSession =
    mononn_engine::tuning::profiler::CuptiProfilingSession;

int main() {
  mononn_engine::tuning::profiler::launch_simple_cuda_kernel(10000);

  CuptiProfilingSession session({"gpu__time_duration.sum"}, 5);

  int counts[]{10000, 10000 * 10, 10000 * 100, 10000 * 1000, 10000 * 2000};

  auto result = session.profiling_context([&]() {
    for (int count : counts) {
      mononn_engine::tuning::profiler::launch_simple_cuda_kernel(count);
    }
  });

  for (int idx = 0; idx < 5; idx++) {
    printf("%f\n", result.get_time_in_us(idx));
  }

  return 0;
}