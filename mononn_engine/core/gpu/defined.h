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

#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace core {
namespace gpu {
struct CUDADefined {
  using CUDAContext = mononn_engine::core::context::CUDAContext;

  static const std::string threadIdx_x;
  static const std::string threadIdx_y;
  static const std::string threadIdx_z;

  static const std::string blockIdx_x;
  static const std::string blockIdx_y;
  static const std::string blockIdx_z;

  static const std::string blockDim_x;
  static const std::string blockDim_y;
  static const std::string blockDim_z;

  static const std::string gridDim_x;
  static const std::string gridDim_y;
  static const std::string gridDim_z;

  static const std::string warpSize;

  static const std::string threadIdx_x_global;
  static const std::string threadCnt_x_global;

  static std::string initialize(const CUDAContext* cuda_context);

  static const std::string warp_block_id;
  static const std::string warp_global_id;
  static const std::string warp_block_count;
  static const std::string warp_global_count;
  static const std::string lane_id;
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine