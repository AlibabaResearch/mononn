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

#include "mononn_engine/core/op_impl/transpose_smem_impl.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/semantic/function_invocation.h"

namespace mononn_engine {
namespace core {
namespace op_impl {
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

std::string TransposeSmemImpl::generate_impl() const {
  auto type = this->output.get_dtype();
  FunctionInvocation invocation("transpose_smem");
  invocation.add_template_arg(type.get_primitive_type().to_string());
  invocation.add_template_arg(std::to_string(this->input_spec.batch_dim));
  invocation.add_template_arg(std::to_string(this->input_spec.dim_r));
  invocation.add_template_arg(std::to_string(this->input_spec.dim_c));
  invocation.add_template_arg(std::to_string(this->smem_tile_dim));
  invocation.add_template_arg(
      BufferManager::get_buffer_name(this->output.get_name()));
  invocation.add_template_arg(
      BufferManager::get_buffer_name(this->input_spec.operand.get_name()));

  return invocation.to_string();
}

std::vector<Tensor> TransposeSmemImpl::get_input_tensor() const {
  return {this->input_spec.operand};
}

std::vector<Tensor> TransposeSmemImpl::get_output_tensor() const {
  return {this->output};
}

int TransposeSmemImpl::get_elements_per_access() const {
  return this->input_spec.operand.get_dtype().get_elements_per_access();
}

void TransposeSmemImpl::set_smem_tile_dim(int _dim) {
  this->smem_tile_dim = _dim;
}

int TransposeSmemImpl::get_smem_tile_dim() const { return this->smem_tile_dim; }

int TransposeSmemImpl::get_smem_usage_in_bytes() const {
  int additional_padding =
      4 / this->output.get_dtype().get_primitive_type().size_in_bytes();
  return this->smem_tile_dim * (this->smem_tile_dim + additional_padding) *
         this->output.get_dtype().get_primitive_type().size_in_bytes();
}

std::vector<std::shared_ptr<OpImplBase>>
TransposeSmemImpl::get_available_implementations(
    std::shared_ptr<CUDAContext> cuda_context, InputSpec input_spec,
    Tensor output) {
  std::vector<int> tile_size_list = {16, 32, 64};

  std::vector<std::shared_ptr<OpImplBase>> impl_list;

  for (auto const& tile_size : tile_size_list) {
    std::shared_ptr<TransposeSmemImpl> impl =
        std::make_shared<TransposeSmemImpl>(cuda_context, input_spec, output);
    impl->set_smem_tile_dim(tile_size);

    impl_list.push_back(std::static_pointer_cast<OpImplBase>(impl));
  }

  return impl_list;
}

std::string TransposeSmemImpl::get_prerequisite_definition() {
  return R"(
template<
  typename T,
  int BlockDim,  // How many threads in Thread Block.
  int BatchCount, // transpose batch
  int Dim_r, // transpose R dimension
  int Dim_c,  // transpose C dimension
  int TileDim = 64>
__global__
void transpose_smem_ilp(T *__restrict__ data_out, T *__restrict__ data_in) {
    extern __shared__ int8_t shared_mem[];

    constexpr int additional_padding = 4 / sizeof(T);
    T (*s_cache)[TileDim + additional_padding] = reinterpret_cast<T (*)[TileDim + additional_padding]>(shared_mem);

    static_assert(BlockDim % TileDim == 0, "Block shape mismatch with tile shape");
    static_assert((TileDim * TileDim) % BlockDim == 0, "Tile shape mismatch with block shape");
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "Unsupported data type");

    constexpr int BLOCK_ROWS = BlockDim / TileDim;
    constexpr int BLOCK_ROW_COUNT = TileDim * TileDim / BlockDim;
    constexpr int TileCount_r = ((Dim_r + TileDim - 1 ) / TileDim);
    constexpr int TileCount_c = ((Dim_c + TileDim - 1 ) / TileDim);
    constexpr int GridTileOneBatch = TileCount_r * TileCount_c;
    constexpr int GridTileTotal = BatchCount * GridTileOneBatch;

    for (int grid_id_total = blockIdx.x ; grid_id_total < GridTileTotal; grid_id_total += gridDim.x) {
        int batch_id = grid_id_total / GridTileOneBatch;
        int grid_id = grid_id_total % GridTileOneBatch;

        int tile_id_r = grid_id / TileCount_c;
        int tile_id_c = grid_id % TileCount_c;

        int batch_offset = batch_id * Dim_r * Dim_c;

        int r_offset_in_tile = threadIdx.x / TileDim;
        int c_offset_in_tile = threadIdx.x % TileDim;
        int r = tile_id_r * TileDim + r_offset_in_tile;
        int c = tile_id_c * TileDim + c_offset_in_tile;

        T input_reg[BLOCK_ROW_COUNT] = {0};

        #pragma unroll
        for (int block_row_offset = 0; block_row_offset < BLOCK_ROW_COUNT; ++block_row_offset) {

            // if (r < Dim_r && c < Dim_c) input_reg[block_row_offset] = data_in[batch_offset + r * Dim_c + c];

            // cutlass::arch::global_load<T, sizeof(T)>(input_reg[block_row_offset], &data_in[batch_offset + r * Dim_c + c], r < Dim_r && c < Dim_c);

            unsigned &data = reinterpret_cast<unsigned &>(input_reg[block_row_offset]);

            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  @p ld.global.u32 %0, [%1];\n"
                "}\n"
                : "=r"(data)
                : "l"(&data_in[batch_offset + r * Dim_c + c]), "r"((int)(r < Dim_r && c < Dim_c)));

            r_offset_in_tile += BLOCK_ROWS;
            r += BLOCK_ROWS;
        }

        asm volatile("barrier.arrive 15, %0;" :: "r"(BlockDim));


        r_offset_in_tile = threadIdx.x / TileDim;
        c_offset_in_tile = threadIdx.x % TileDim;
        r = tile_id_r * TileDim + r_offset_in_tile;
        c = tile_id_c * TileDim + c_offset_in_tile;

        #pragma unroll
        for (int block_row_offset = 0; block_row_offset < BLOCK_ROW_COUNT; ++block_row_offset) {

            if (r < Dim_r && c < Dim_c) s_cache[r_offset_in_tile][c_offset_in_tile] = input_reg[block_row_offset];

            r_offset_in_tile += BLOCK_ROWS;
            r += BLOCK_ROWS;
        }

        __syncthreads();

        r_offset_in_tile = threadIdx.x / TileDim;
        c_offset_in_tile = threadIdx.x % TileDim;
        r = tile_id_c * TileDim + r_offset_in_tile;
        c = tile_id_r * TileDim + c_offset_in_tile;

        #pragma unroll
        for (int block_row_offset = 0; block_row_offset < BLOCK_ROW_COUNT; ++block_row_offset) {
            if (r < Dim_c && c < Dim_r) input_reg[block_row_offset] = s_cache[c_offset_in_tile][r_offset_in_tile];
            r_offset_in_tile += BLOCK_ROWS;
            r += BLOCK_ROWS;
        }

        r_offset_in_tile = threadIdx.x / TileDim;
        c_offset_in_tile = threadIdx.x % TileDim;
        r = tile_id_c * TileDim + r_offset_in_tile;
        c = tile_id_r * TileDim + c_offset_in_tile;

        #pragma unroll
        for (int block_row_offset = 0; block_row_offset < BLOCK_ROW_COUNT; ++block_row_offset) {
            if (r < Dim_c && c < Dim_r) data_out[batch_offset + r * Dim_r + c] = input_reg[block_row_offset];
            r_offset_in_tile += BLOCK_ROWS;
            r += BLOCK_ROWS;
        }

        __syncthreads();
    }
}
)";
}

void TransposeSmemImpl::set_instruction_parallel_factor(int _ilp_factor) {
  LOG(FATAL) << "Unimplemented";
}
}  // namespace op_impl
}  // namespace core
}  // namespace mononn_engine