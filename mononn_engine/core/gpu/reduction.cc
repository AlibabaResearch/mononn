#include "mononn_engine/core/gpu/reduction.h"

#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/macros.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace gpu {

std::vector<std::string> Reduction::get_implementations() {
  return {"warp", "block", "block_cub_BLOCK_REDUCE_RAKING",
          "block_cub_BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
          "block_cub_BLOCK_REDUCE_WARP_REDUCTIONS"};
}

std::string Reduction::get_op_definition() {
  return R"(
template<typename T, int N, typename ReductionOp>
__device__ __forceinline__
T to_scalar(cutlass::Array<T, N> &array) {
    ReductionOp op;
    T ret = array[0];

    #pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return ret;
}

template<typename Element, typename ReductionOp, int WarpSize>
__device__ __forceinline__
Element warp_custom_reduce(Element my_val) {
    ReductionOp op;
    Element ret = my_val;

    #pragma unroll
    for (int offset = (WarpSize >> 1); offset > 0; offset >>= 1) {
        ret  = op(ret, __shfl_down_sync(0xffffffff, ret, offset));
    }

    return ret;
}

template<typename Element, typename ReductionOp, int WarpSize>
__device__ __forceinline__
Element warp_reduce_cub(Element my_val) {
    extern __shared__ Element s_cache[];

    using WarpReduce = cub::WarpReduce<Element, WarpSize, 800>;
    typename WarpReduce::TempStorage *temp_storage = reinterpret_cast<typename WarpReduce::TempStorage *>(s_cache);
    return WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize>
__device__ __forceinline__
Element block_custom_reduce(Element my_val) {
    ReductionOp op;
    extern __shared__ Element s_cache[];
    Element ret = my_val;

    s_cache[threadIdx.x] = ret;
    __syncthreads();

    #pragma unroll
    for (int offset = (BlockSize >> 1); offset > 0; offset >>= 1) {
        Element tmp;
        if (threadIdx.x + offset < blockDim.x) {
            tmp = s_cache[threadIdx.x + offset];
        }
        
        __syncthreads();

        if (threadIdx.x + offset < blockDim.x) {
            s_cache[threadIdx.x] = op(s_cache[threadIdx.x], tmp);
        }

        __syncthreads();
    }

    ret = s_cache[threadIdx.x];

    __syncthreads();

    return ret;
}


template<typename Element, typename ReductionOp, int BlockSize>
__device__ __forceinline__
Element block_reduce_cub_RAKING(Element my_val) {
    extern __shared__ Element s_cache[];
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, __CUDA_ARCH_GLOBAL__>;

    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize>
__device__ __forceinline__
Element block_reduce_cub_RAKING_COMMUTATIVE_ONLY(Element my_val) {
    extern __shared__ Element s_cache[];
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize>
__device__ __forceinline__
Element block_reduce_cub_WARP_REDUCTIONS(Element my_val) {
    extern __shared__ Element s_cache[];
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

        )";
}

using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using Functor = mononn_engine::core::gpu::Functor;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using Tensor = mononn_engine::core::tensor::Tensor;
std::vector<Reduction::FunctionInvocation> Reduction::get_invocations(
    std::shared_ptr<CUDAContext> context, const Tensor& operand,
    const Tier& tier, const Functor& reducer) {
  std::vector<Reduction::FunctionInvocation> invocations;
  bool vectorized_mem_access =
      operand.get_dtype().get_elements_per_access() > 1;
  Reduction::FunctionInvocation vector_to_scalar("to_scalar");
  if (vectorized_mem_access) {
    vector_to_scalar.add_template_arg(
        operand.get_dtype().get_primitive_type().to_string());
    vector_to_scalar.add_template_arg(
        std::to_string(operand.get_dtype().get_elements_per_access()));
    vector_to_scalar.add_template_arg(reducer.get_functor_type());
    vector_to_scalar.add_arg(operand.get_name());
  }

  if (tier == LocalityTier::kT0) {
    LOG(WARNING) << "Unsupported tier t0";
    return {};
  }

  if (tier == LocalityTier::kT1) {
    Reduction::FunctionInvocation warp_custom_reduce("warp_custom_reduce");
    warp_custom_reduce.add_template_arg(operand.get_dtype().to_string());
    warp_custom_reduce.add_template_arg(reducer.get_functor_type());
    warp_custom_reduce.add_template_arg(
        std::to_string(context->cuda_device_context.warp_size));

    if (vectorized_mem_access)
      warp_custom_reduce.add_arg(vector_to_scalar.to_string());
    else
      warp_custom_reduce.add_arg(operand.get_name());

    Reduction::FunctionInvocation warp_reduce_cub("warp_reduce_cub");

    warp_reduce_cub.add_template_arg(operand.get_dtype().to_string());
    warp_reduce_cub.add_template_arg(reducer.get_functor_type());
    warp_reduce_cub.add_template_arg(
        std::to_string(context->cuda_device_context.warp_size));

    if (vectorized_mem_access)
      warp_reduce_cub.add_arg(vector_to_scalar.to_string());
    else
      warp_reduce_cub.add_arg(operand.get_name());

    return {warp_custom_reduce, warp_reduce_cub};
  }

  if (tier == LocalityTier::kT2) {
    Reduction::FunctionInvocation block_custom_reduce("block_custom_reduce");
    Reduction::FunctionInvocation block_reduce_cub_RAKING(
        "block_reduce_cub_RAKING");
    Reduction::FunctionInvocation block_reduce_cub_RAKING_COMMUTATIVE_ONLY(
        "block_reduce_cub_RAKING_COMMUTATIVE_ONLY");
    Reduction::FunctionInvocation block_reduce_cub_WARP_REDUCTIONS(
        "block_reduce_cub_WARP_REDUCTIONS");

    std::vector<Reduction::FunctionInvocation> ret = {
        block_custom_reduce, block_reduce_cub_RAKING,
        block_reduce_cub_RAKING_COMMUTATIVE_ONLY,
        block_reduce_cub_WARP_REDUCTIONS};

    for (auto& reduce_invocation : ret) {
      reduce_invocation.add_template_arg(operand.get_dtype().to_string());
      reduce_invocation.add_template_arg(reducer.get_functor_type());
      reduce_invocation.add_template_arg(
          std::to_string(context->cuda_runtime_context.block_dim.x));

      if (vectorized_mem_access)
        reduce_invocation.add_arg(vector_to_scalar.to_string());
      else
        reduce_invocation.add_arg(operand.get_name());
    }

    return ret;
  }

  if (tier == LocalityTier::kT3) {
    LOG(WARNING) << "Unsupported tier t3";
    return {};
  }

  LOG(WARNING) << "Unexpected locality tier: " << tier.to_string();
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine