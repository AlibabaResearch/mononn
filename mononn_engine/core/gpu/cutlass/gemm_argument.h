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

#include "mononn_engine/core/gpu/cutlass/gemm_coord.h"
#include "mononn_engine/core/gpu/cutlass/gemm_universal_mode.h"
namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct GemmUniversalArgument {
  GemmUniversalMode mode;
  GemmCoord problem_size;
  std::string batch_count;
  std::string alpha, beta;
  std::string ptr_A;
  std::string ptr_B;
  std::string ptr_C;
  std::string ptr_D;
  std::string batch_stride_A;
  std::string batch_stride_B;
  std::string batch_stride_C;
  std::string batch_stride_D;
  std::string stride_a;
  std::string stride_b;
  std::string stride_c;
  std::string stride_d;

  std::string define_variable(std::string gemm_kernel,
                              std::string var_name) const;
};

struct GemmWithLoopFusionArgument {};

struct GemmWithInputFusionArgument {};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine