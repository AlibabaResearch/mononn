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

#include "mononn_engine/core/context/cuda_utils.h"

#include <cuda_runtime.h>

#include <iostream>

#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace context {

static const char* _cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void mock_kernel() {}

// On A100
// TB/SM| max per block sm size | per sm reserved sm size
// 1: 166912 1024
// 2: 82944 2048
// 3: 54912 3200
// 4: 40960 4096
// 5: 32512 5376
// 6: 26880 6656

int get_max_smem_size_per_block(int desired_block_count_per_sm, int block_size,
                                int max_configurable_smem,
                                int reserved_smem_per_block) {
  checkCudaErrors(cudaFuncSetAttribute(
      mock_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_configurable_smem));
  size_t dynamicSmemSize;
  checkCudaErrors(cudaOccupancyAvailableDynamicSMemPerBlock(
      &dynamicSmemSize, mock_kernel, desired_block_count_per_sm, block_size));

  if (desired_block_count_per_sm > 1) {
    dynamicSmemSize -= reserved_smem_per_block;
  }

  int numBlocks;

  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, mock_kernel, block_size, dynamicSmemSize));

  if (numBlocks != desired_block_count_per_sm) {
    LOG(FATAL) << "Block count mismatch, desired block per sm "
               << desired_block_count_per_sm << " got " << numBlocks << " with "
               << dynamicSmemSize << " bytes dynamic smem per block.";
  }

  // Check if return max available smem size.
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, mock_kernel, block_size, dynamicSmemSize + 1));

  if (numBlocks != desired_block_count_per_sm - 1) {
    LOG(FATAL) << "Does not reach max smem size per block. Got "
               << dynamicSmemSize << " bytes dynamic smem when block per sm is "
               << desired_block_count_per_sm;
  }

  return dynamicSmemSize;
  // int L = 0, R = max_configurable_smem;

  // while (L <= R) {
  //     int M = L + (R - L) / 2;
  //     int numBlocks;

  //     checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  //             &numBlocks,
  //             mock_kernel,
  //             block_size,
  //             M));

  //     if (numBlocks < desired_block_count_per_sm) R = M - 1;
  //     else L = M + 1;
  // }

  // return R;
}
}  // namespace context
}  // namespace core
}  // namespace mononn_engine
