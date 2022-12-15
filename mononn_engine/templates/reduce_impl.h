#pragma once 
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/functional_addon.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_profiler_api.h>
#define __CUDA_ARCH_GLOBAL__ 800

template<typename Element, typename ReductionOp, int WarpSize, int N>
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

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
cutlass::AlignedArray<Element, N> warp_custom_reduce(cutlass::AlignedArray<Element, N> &my_val) {
    ReductionOp op;
    cutlass::AlignedArray<Element, N> ret = my_val;

    #pragma unroll
    for (int offset = (WarpSize >> 1); offset > 0; offset >>= 1) {
        #pragma unroll 
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = op(ret[idx], __shfl_down_sync(0xffffffff, ret[idx], offset));
        }
    }

    return ret;
}

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
Element warp_reduce_cub(Element my_val) {
    static_assert(WarpSize == 32, "Warp size should use 32"); // Power of 2 warp size will not use shared memory.
    extern __shared__ int8_t __cache[];
    Element *s_cache = reinterpret_cast<Element *>(__cache);

    using WarpReduce = cub::WarpReduce<Element, WarpSize, __CUDA_ARCH_GLOBAL__>;
    typename WarpReduce::TempStorage *temp_storage = reinterpret_cast<typename WarpReduce::TempStorage *>(s_cache);
    return WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int WarpSize, int N>
__device__ __forceinline__
cutlass::AlignedArray<Element, N> warp_reduce_cub(cutlass::AlignedArray<Element, N> &my_val) {
    static_assert(WarpSize == 32, "Warp size should use 32"); // Power of 2 warp size will not use shared memory.

    extern __shared__ int8_t __cache[];
    Element *s_cache = reinterpret_cast<Element *>(__cache);

    using WarpReduce = cub::WarpReduce<Element, WarpSize, __CUDA_ARCH_GLOBAL__>;
    typename WarpReduce::TempStorage *temp_storage = reinterpret_cast<typename WarpReduce::TempStorage *>(s_cache);

    cutlass::AlignedArray<Element, N> result;

    for (int idx = 0; idx < N; ++idx) {
        result[idx] = WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(my_val[idx], ReductionOp());
    }

    return result;
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_custom_reduce(Element my_val, void *__cache) {
    const int lane_id = threadIdx.x & 0x1f;
    const int warp_id = threadIdx.x >> 5;

    Element *s_cache = reinterpret_cast<Element *>(__cache);

    Element ret = my_val;
    ret = warp_custom_reduce<Element, ReductionOp, 32, 1>(ret);

    if (lane_id == 0) 
        s_cache[warp_id] = ret;
    __syncthreads();
    
    ret = (threadIdx.x < blockDim.x / 32) ? s_cache[lane_id] : Element(0);
    ret = warp_custom_reduce<Element, ReductionOp, 32, 1>(ret);
    return ret;
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_RAKING(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_RAKING_COMMUTATIVE_ONLY(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}

template<typename Element, typename ReductionOp, int BlockSize, int N>
__device__ __forceinline__
Element block_reduce_cub_WARP_REDUCTIONS(Element my_val, void *__cache) {
    Element *s_cache = reinterpret_cast<Element *>(__cache);
    using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, __CUDA_ARCH_GLOBAL__>;
    typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);
    return BlockReduce(*temp_storage).Reduce(my_val, ReductionOp());
}


template<typename T, int N, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const cutlass::Array<T, N> &array) {
    ReductionOp op;
    T ret = array[0];

    #pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return ret;
}

template<typename T, int N, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const cutlass::AlignedArray<T, N> &array) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return ret;
}

template<typename T, typename ReductionOp>
__device__ __forceinline__
T to_scalar(const T &val) {
    return val;
}

template<typename T, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const T &val, TArgs &... args) {
    ReductionOp op;
    return op(val, to_scalar<T, ReductionOp>(args...));
}

template<typename T, int N, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const cutlass::Array<T, N> &array, TArgs &... args) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return op(ret, to_scalar<T, N, ReductionOp>(args...));
}

template<typename T, int N, typename ReductionOp, typename ... TArgs>
__device__ __forceinline__
T to_scalar(const cutlass::AlignedArray<T, N> &array, TArgs &... args) {
    ReductionOp op;
    T ret = array[0];

#pragma unroll
    for (int iter = 1; iter < N; ++iter) {
        ret = op(ret, array[iter]);
    }

    return op(ret, to_scalar<T, N, ReductionOp>(args...));
}

template<typename T>
__device__ __forceinline__
T warp_broadcast(const T &val) {
    return __shfl_sync(0xffffffff, val, 0);
}

template<typename T>
__device__ __forceinline__
T block_broadcast(const T &val, void *__cache) {
    T *s_cache = reinterpret_cast<T *>(__cache);
    if (threadIdx.x == 0) {
        s_cache[0] = val;
    }

    __syncthreads();

    return s_cache[0];
}


// To use multi reduction, we need to re-organize shared memory planning to ensure access to shared memory will not out of bound.
// template<typename Element, typename ReductionOp, int BlockSize, int N>
// __device__ __forceinline__
// cutlass::AlignedArray<Element, N> block_custom_reduce(cutlass::AlignedArray<Element, N> &my_val, cutlass::AlignedArray<Element, N> *s_cache) {
//     const int lane_id = threadIdx.x & 0x1f;
//     const int warp_id = threadIdx.x >> 5;

//     cutlass::AlignedArray<Element, N> ret = my_val;
//     ret = warp_custom_reduce<Element, ReductionOp, 32, N>(ret);

//     if (lane_id == 0) 
//         s_cache[warp_id] = ret;
//     __syncthreads();
    
//     ret = (threadIdx.x < blockDim.x / 32) ? s_cache[lane_id] : Element(0);
//     ret = warp_custom_reduce<Element, ReductionOp, 32, N>(ret);
//     return ret;
// }

// template<typename Element, typename ReductionOp, int BlockSize, int N>
// __device__ __forceinline__
// cutlass::AlignedArray<Element, N> block_reduce_cub_RAKING(cutlass::AlignedArray<Element, N> my_val) {
//     extern __shared__ int8_t __cache[];
//     Element *s_cache = reinterpret_cast<Element *>(__cache);

//     using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, __CUDA_ARCH_GLOBAL__>;

//     typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);

//     cutlass::AlignedArray<Element, N> result;
    
//     for (int idx = 0; idx < N; ++idx) {
//         result[idx] = BlockReduce(*temp_storage).Reduce(my_val[idx], ReductionOp());
//     }

//     return result;
// }

// template<typename Element, typename ReductionOp, int BlockSize, int N>
// __device__ __forceinline__
// cutlass::AlignedArray<Element, N> block_reduce_cub_RAKING_COMMUTATIVE_ONLY(cutlass::AlignedArray<Element, N> my_val) {
//     extern __shared__ int8_t __cache[];
//     Element *s_cache = reinterpret_cast<Element *>(__cache);

//     using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH_GLOBAL__>;
//     typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);

//     cutlass::AlignedArray<Element, N> result;

//     for (int idx = 0; idx < N; ++idx) {
//         result[idx] = BlockReduce(*temp_storage).Reduce(my_val[idx], ReductionOp());
//     }
// }

// template<typename Element, typename ReductionOp, int BlockSize, int N>
// __device__ __forceinline__
// cutlass::AlignedArray<Element, N> block_reduce_cub_WARP_REDUCTIONS(cutlass::AlignedArray<Element, N> my_val) {
//     extern __shared__ int8_t __cache[];
//     Element *s_cache = reinterpret_cast<Element *>(__cache);

//     using BlockReduce = cub::BlockReduce<Element, BlockSize, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, __CUDA_ARCH_GLOBAL__>;
//     typename BlockReduce::TempStorage *temp_storage = reinterpret_cast<typename BlockReduce::TempStorage *>(s_cache);

//     cutlass::AlignedArray<Element, N> result;

//     for (int idx = 0; idx < N; ++idx) {
//         result[idx] = BlockReduce(*temp_storage).Reduce(my_val[idx], ReductionOp());
//     }
// }