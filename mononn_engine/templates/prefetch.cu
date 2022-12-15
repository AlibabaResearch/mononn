#include <cuda_runtime.h>
#include <cstdio>

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

// #include <cub/cub.cuh>

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

__launch_bounds__(128, 2)
__global__
void FMA_kernel(float *A, float *B, float *C, float *D, int count) {
    extern __shared__ int8_t __cache[];
    asynchronous::copy<16, 16>()((void *)__cache, A, true);

    constexpr int unroll_factor = 16;
    // #pragma unroll(1)
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < count; idx += blockDim.x * gridDim.x * unroll_factor) {
        #pragma unroll(unroll_factor) 
        for (int i = 0; i < unroll_factor; ++i) {
            D[idx + i * blockDim.x * gridDim.x] = A[idx + i * blockDim.x * gridDim.x] * B[idx + i * blockDim.x * gridDim.x] + C[idx + i * blockDim.x * gridDim.x];
        }
    }
}

int main(int argc, char const *argv[])
{
    int count = 108*2*128*1024;
    float *a, *b, *c, *d;
    checkCudaErrors(cudaMalloc(&a, sizeof(float) * count));
    checkCudaErrors(cudaMalloc(&b, sizeof(float) * count));
    checkCudaErrors(cudaMalloc(&c, sizeof(float) * count));
    checkCudaErrors(cudaMalloc(&d, sizeof(float) * count));

    FMA_kernel<<<108*2, 128>>>(a, b, c, d, count);

    checkCudaErrors(cudaDeviceSynchronize());

    // using WarpReduce = cub::WarpReduce<float, 32, 800>;
    // printf("%d\n", (int)sizeof(typename WarpReduce::TempStorage));

    // using BlockReduce1 = cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, 800>;
    // using BlockReduce2 = cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, 800>;
    // using BlockReduce3 = cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 800>;

    // printf("%d\n", (int)sizeof(typename BlockReduce1::TempStorage));
    // printf("%d\n", (int)sizeof(typename BlockReduce2::TempStorage));
    // printf("%d\n", (int)sizeof(typename BlockReduce3::TempStorage));

    return 0;
}
