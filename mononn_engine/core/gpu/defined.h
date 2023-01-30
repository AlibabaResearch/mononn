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