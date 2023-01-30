#include "mononn_engine/core/gpu/headers.h"

namespace mononn_engine {
namespace core {
namespace gpu {
std::string Headers::get_headers() {
  return R"(
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cuda/std/tuple>

#include "cutlass/functional.h"
#include "cutlass/functional_addon.h"
)";
}

std::string Headers::get_headers_main_only() {
  return R"(
#include <array>

#include "cnpy/cnpy.h"
#include "cutlass/arch/memory.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
)";
}

std::string Headers::get_cuda_helpers() {
  return R"(
static const char *_cudaGetErrorEnum(cudaError_t error) {
        return cudaGetErrorName(error);
    }

template <typename T>
void check(T result, char const *const func, const char *const file,
        int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T, typename TMi, typename Tma>
__device__ __forceinline__
T clamp(const T &value, const TMi &min_value, const Tma &max_value) {
    return min((T)max_value, max(value, (T)min_value));
}
)";
}

std::string Headers::get_async_copy_headers() {
  return R"(
namespace asynchronous {

template<int kSizeInBytes, int kSrcInBytes>
struct copy {
    static_assert(kSizeInBytes == 4 || kSizeInBytes == 8 || kSizeInBytes == 16);
    __device__ __forceinline__
    void operator () (void *smem_ptr, void *global_ptr, bool pred) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
            "}\n"
            :
            : "r"((int)pred), "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))), "l"(global_ptr),
                "n"(kSizeInBytes), "n"(kSrcInBytes)
            );
    }
};

template<int kSizeInBytes>
struct copy<kSizeInBytes, kSizeInBytes> {
    static_assert(kSizeInBytes == 4 || kSizeInBytes == 8);
    __device__ __forceinline__
    void operator () (void *smem_ptr, void *global_ptr, bool pred) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
            "}\n"
            :
            : "r"((int)pred), "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))), "l"(global_ptr),
                "n"(kSizeInBytes)
            );
    }
};

template<>
struct copy<16, 16> {
    __device__ __forceinline__
    void operator () (void *smem_ptr, void *global_ptr, bool pred) {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
            "}\n"
            :
            : "r"((int)pred), "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))), "l"(global_ptr),
                "n"(16)
            );
    }
};

struct commit {
    __device__ __forceinline__
    void operator() () {
        asm volatile("cp.async.commit_group;\n" ::);
    }
};

template<int N>
struct wait {
    __device__ __forceinline__
    void operator() () {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
    }
};

struct wait_all {
    __device__ __forceinline__
    void operator() () {
        asm volatile("cp.async.wait_all;\n" ::);
    }
};
}
        )";
}

std::string Headers::get_tuple_headers() {
  return R"(

namespace tuple_util {
    template<typename ... TArgs>
    __device__ __forceinline__
    cuda::std::tuple<TArgs...> make_tuple(const TArgs &... args) {
        return cuda::std::make_tuple(args...);
    }

    template<typename T1, typename T2, int N>
    __device__ __forceinline__
    cutlass::AlignedArray<cuda::std::tuple<T1, T2>, N> make_tuple(const cutlass::AlignedArray<T1, N> &a, const cutlass::AlignedArray<T2, N> &b) {
        cutlass::AlignedArray<cuda::std::tuple<T1, T2>, N> ret;
        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = cuda::std::make_tuple(a[idx], b[idx]);
        }

        return ret;
    }

    template<typename T1, typename T2, typename T3, int N>
    __device__ __forceinline__
    cutlass::AlignedArray<cuda::std::tuple<T1, T2, T3>, N> make_tuple(const cutlass::AlignedArray<T1, N> &a, const cutlass::AlignedArray<T2, N> &b, const cutlass::AlignedArray<T3, N> &c) {
        cutlass::AlignedArray<cuda::std::tuple<T1, T2, T3>, N> ret;
        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = cuda::std::make_tuple(a[idx], b[idx], c[idx]);
        }

        return ret;
    }

    template<typename T1, typename T2, int N>
    __device__ __forceinline__
    cutlass::Array<cuda::std::tuple<T1, T2>, N> make_tuple(const cutlass::Array<T1, N> &a, const cutlass::Array<T2, N> &b) {
        cutlass::Array<cuda::std::tuple<T1, T2>, N> ret;
        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = cuda::std::make_tuple(a[idx], b[idx]);
        }

        return ret;
    }

    template<typename T1, typename T2, typename T3, int N>
    __device__ __forceinline__
    cutlass::Array<cuda::std::tuple<T1, T2, T3>, N> make_tuple(const cutlass::Array<T1, N> &a, const cutlass::Array<T2, N> &b, const cutlass::Array<T3, N> &c) {
        cutlass::Array<cuda::std::tuple<T1, T2, T3>, N> ret;
        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = cuda::std::make_tuple(a[idx], b[idx], c[idx]);
        }

        return ret;
    }
}
        )";
}

std::string Headers::get_tuple_shfl_headers() {
  return R"(
template<typename T1, typename T2>
__device__ __forceinline__
cuda::std::tuple<T1, T2> __shfl_sync(unsigned mask, const cuda::std::tuple<T1, T2> &val, int srcLane) {
    return cuda::std::make_tuple(
            __shfl_sync(mask, cuda::std::get<0>(val), srcLane), 
            __shfl_sync(mask, cuda::std::get<1>(val), srcLane));
}

template<typename T1, typename T2, typename T3>
__device__ __forceinline__
cuda::std::tuple<T1, T2, T3> __shfl_sync(unsigned mask, const cuda::std::tuple<T1, T2, T3> &val, int srcLane) {
    return cuda::std::make_tuple(
            __shfl_sync(mask, cuda::std::get<0>(val), srcLane), 
            __shfl_sync(mask, cuda::std::get<1>(val), srcLane),
            __shfl_sync(mask, cuda::std::get<2>(val), srcLane));
}

template<typename T1, typename T2>
__device__ __forceinline__
cuda::std::tuple<T1, T2> __shfl_down_sync(unsigned mask, const cuda::std::tuple<T1, T2> &val, int delta) {
    return cuda::std::make_tuple(
            __shfl_down_sync(mask, cuda::std::get<0>(val), delta), 
            __shfl_down_sync(mask, cuda::std::get<1>(val), delta));
}

template<typename T1, typename T2, typename T3>
__device__ __forceinline__
cuda::std::tuple<T1, T2, T3> __shfl_down_sync(unsigned mask, const cuda::std::tuple<T1, T2, T3> &val, int delta) {
    return cuda::std::make_tuple(
            __shfl_down_sync(mask, cuda::std::get<0>(val), delta), 
            __shfl_down_sync(mask, cuda::std::get<1>(val), delta),
            __shfl_down_sync(mask, cuda::std::get<2>(val), delta));
}
        )";
}
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine