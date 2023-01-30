#include "mononn_engine/core/gpu/defined.h"

#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
const std::string CUDADefined::threadIdx_x = "threadIdx.x";
const std::string CUDADefined::threadIdx_y = "threadIdx.y";
const std::string CUDADefined::threadIdx_z = "threadIdx.z";

const std::string CUDADefined::blockIdx_x = "blockIdx.x";
const std::string CUDADefined::blockIdx_y = "blockIdx.y";
const std::string CUDADefined::blockIdx_z = "blockIdx.z";

const std::string CUDADefined::blockDim_x = "BlockDim_X";
const std::string CUDADefined::blockDim_y = "BlockDim_Y";
const std::string CUDADefined::blockDim_z = "BlockDim_Z";

const std::string CUDADefined::gridDim_x = "GridDim_X";
const std::string CUDADefined::gridDim_y = "GridDim_Y";
const std::string CUDADefined::gridDim_z = "GridDim_Z";

const std::string CUDADefined::warpSize = "WarpSize";

const std::string CUDADefined::threadIdx_x_global =
    mononn_engine::helpers::string_format(
        "(%s + %s * %s)", CUDADefined::threadIdx_x.c_str(),
        CUDADefined::blockIdx_x.c_str(), CUDADefined::blockDim_x.c_str());
const std::string CUDADefined::threadCnt_x_global =
    mononn_engine::helpers::string_format("(%s * %s)",
                                          CUDADefined::blockDim_x.c_str(),
                                          CUDADefined::gridDim_x.c_str());

std::string CUDADefined::initialize(const CUDAContext* cuda_context) {
  return mononn_engine::helpers::string_format(
      R"(
constexpr int BlockDim_X = %s;
constexpr int BlockDim_Y = %s;
constexpr int BlockDim_Z = %s;
constexpr int GridDim_X = %s;
constexpr int GridDim_Y = %s;
constexpr int GridDim_Z = %s;

constexpr int WarpSize = 32;
const int %s = %s / %s;
const int %s = %s / %s;
const int %s = %s + %s * %s;
const int %s = %s * %s;
const int %s = %s %% %s;
        )",
      std::to_string(cuda_context->cuda_runtime_context.block_dim.x).c_str(),
      std::to_string(cuda_context->cuda_runtime_context.block_dim.y).c_str(),
      std::to_string(cuda_context->cuda_runtime_context.block_dim.z).c_str(),
      std::to_string(cuda_context->cuda_runtime_context.grid_dim.x).c_str(),
      std::to_string(cuda_context->cuda_runtime_context.grid_dim.y).c_str(),
      std::to_string(cuda_context->cuda_runtime_context.grid_dim.z).c_str(),
      CUDADefined::warp_block_id.c_str(), CUDADefined::threadIdx_x.c_str(),
      CUDADefined::warpSize.c_str(), CUDADefined::warp_block_count.c_str(),
      CUDADefined::blockDim_x.c_str(), CUDADefined::warpSize.c_str(),
      CUDADefined::warp_global_id.c_str(), CUDADefined::warp_block_id.c_str(),
      CUDADefined::blockIdx_x.c_str(), CUDADefined::warp_block_count.c_str(),
      CUDADefined::warp_global_count.c_str(), CUDADefined::gridDim_x.c_str(),
      CUDADefined::warp_block_count.c_str(), CUDADefined::lane_id.c_str(),
      CUDADefined::threadIdx_x.c_str(), CUDADefined::warpSize.c_str());
}

const std::string CUDADefined::warp_block_id = "warp_block_id";
const std::string CUDADefined::warp_global_id = "warp_global_id";
const std::string CUDADefined::warp_block_count = "warp_block_count";
const std::string CUDADefined::warp_global_count = "warp_global_count";
const std::string CUDADefined::lane_id = "lane_id";
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine