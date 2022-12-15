#include "reduce_impl.h"
#include "cutlass/arch/memory.h"

__device__ __forceinline__
float4 operator + (const float4 &a, const float4 &b) {
    float4 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    res.z = a.z + b.z;
    res.w = a.w + b.w;
    return res;
}


__device__ __forceinline__
float4& operator += (float4 &a, const float4 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}


#include <cutlass/functional.h>
#include <cub/cub.cuh>



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


template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_warp(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    const int warp_in_block = blockDim.x / warpSize;
    const int warp_id = threadIdx.x / warpSize + blockIdx.x * warp_in_block;
    const int lane_id = threadIdx.x % warpSize;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = warp_id; r < R; r += gridDim.x * warp_in_block) { // schedule 1
        TVec val = init_val; // compute at schedule 1
        for (int c = lane_id ; c < C / vec_len; c += warpSize) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            // TVec element;
            // cutlass::arch::global_load<TVec, sizeof(TVec)>(element, reinterpret_cast<void *>(operand + r * C + c * vec_len), true);

            val = vector_reducer(val, element); // compute at schedule 2
        }

        T ret = warp_custom_reduce<T, ScalarReducer, 32, 1>(to_scalar<T, vec_len, ScalarReducer>(val)); // compute at schedule 1
        if (lane_id == 0) {
            result[r] = ret; // compute as schedule 1
        }
    }
}

// template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
// __global__
// void reduce_warp_test(T *operand, T *result, T init_val, int R, int C) {
//     VectorReducer vector_reducer;

//     const int warp_in_block = blockDim.x / warpSize;
//     const int warp_id = threadIdx.x / warpSize + blockIdx.x * warp_in_block;
//     const int lane_id = threadIdx.x % warpSize;

//     constexpr int vec_len = sizeof(TVec) / sizeof(T);
//     using Tvec_unaligned = cutlass::Array<T, vec_len>;

//     for (int r = warp_id; r < R; r += gridDim.x * warp_in_block) { // schedule 1
//         Tvec_unaligned val = init_val; // compute at schedule 1
//         for (int c = lane_id ; c < C / vec_len; c += warpSize) { // schedule 2
//             Tvec_unaligned element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
//             val = vector_reducer(val, element); // compute at schedule 2
//         }

//         T ret = warp_custom_reduce<T, ScalarReducer>(to_scalar<T, vec_len, ScalarReducer>(val)); // compute at schedule 1
//         if (lane_id == 0) {
//             result[r] = ret; // compute as schedule 1
//         }
//     }
// }

template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_warp_cub(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    const int warp_in_block = blockDim.x / warpSize;
    const int warp_id = threadIdx.x / warpSize + blockIdx.x * warp_in_block;
    const int lane_id = threadIdx.x % warpSize;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = warp_id; r < R; r += gridDim.x * warp_in_block) { // schedule 1
        TVec val = init_val; // compute at schedule 1
        for (int c = lane_id ; c < C / vec_len; c += warpSize) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        // T ret = WarpReduce(temp_storage[threadIdx.x / 32]).Reduce(to_scalar<T, vec_len, ScalarReducer>(val), ScalarReducer());
        T ret = warp_reduce_cub<T, ScalarReducer, 32, 1>(to_scalar<T, vec_len, ScalarReducer>(val));
        // T ret = warp_custom_reduce<T, ScalarReducer, 16>(0xffffffff, to_scalar<T, vec_len, ScalarReducer>(val)); // compute at schedule 1

        if (lane_id == 0) {
            result[r] = ret; // compute as schedule 1
        }
    }
}


template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_block(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = blockIdx.x; r < R; r += gridDim.x) { // schedule 1
        TVec val = init_val;
        for (int c = threadIdx.x ; c < C / vec_len; c += blockDim.x) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        T ret = block_custom_reduce<T, ScalarReducer, 128, 1>(to_scalar<T, vec_len, ScalarReducer>(val)); // compute at schedule 1
        if (threadIdx.x == 0) {
            result[r] = ret; // compute as schedule 1
            // result[r] = val.x + val.y + val.z + val.w;
        }
    }
}


template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_block_cub_RAKING(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = blockIdx.x; r < R; r += gridDim.x) { // schedule 1
        TVec val = init_val;
        for (int c = threadIdx.x ; c < C / vec_len; c += blockDim.x) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        // T ret = BlockReduce(*temp_storage).Reduce(to_scalar<T, vec_len, ScalarReducer>(val), ScalarReducer());
        T ret = block_reduce_cub_RAKING<T, ScalarReducer, 128, 1>(to_scalar<T, vec_len, ScalarReducer>(val));
        
        if (threadIdx.x == 0) {
            result[r] = ret; // compute as schedule 1
            // result[r] = val.x + val.y + val.z + val.w;
        }
    }
}

template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_block_cub_RAKING_COMMUTATIVE_ONLY(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = blockIdx.x; r < R; r += gridDim.x) { // schedule 1
        TVec val = init_val;
        for (int c = threadIdx.x ; c < C / vec_len; c += blockDim.x) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        // T ret = BlockReduce(*temp_storage).Reduce(to_scalar<T, vec_len, ScalarReducer>(val), ScalarReducer());
        T ret = block_reduce_cub_RAKING_COMMUTATIVE_ONLY<T, ScalarReducer, 128, 1>(to_scalar<T, vec_len, ScalarReducer>(val));
        
        if (threadIdx.x == 0) {
            result[r] = ret; // compute as schedule 1
            // result[r] = val.x + val.y + val.z + val.w;
        }
    }
}

template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__global__
void reduce_block_cub_WARP_REDUCTIONS(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = blockIdx.x; r < R; r += gridDim.x) { // schedule 1
        TVec val = init_val;
        for (int c = threadIdx.x ; c < C / vec_len; c += blockDim.x) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        // T ret = BlockReduce(*temp_storage).Reduce(to_scalar<T, vec_len, ScalarReducer>(val), ScalarReducer());
        T ret = block_reduce_cub_WARP_REDUCTIONS<T, ScalarReducer, 128, 1>(to_scalar<T, vec_len, ScalarReducer>(val));
        
        if (threadIdx.x == 0) {
            result[r] = ret; // compute as schedule 1
            // result[r] = val.x + val.y + val.z + val.w;
        }
    }
}

template<typename T, typename TVec, typename ScalarReducer, typename VectorReducer>
__launch_bounds__(128, 4)
__global__
void reduce_warp_ILP(T *operand, T *result, T init_val, int R, int C) {
    VectorReducer vector_reducer;

    const int warp_in_block = blockDim.x / warpSize;
    const int warp_id = threadIdx.x / warpSize + blockIdx.x * warp_in_block;
    const int lane_id = threadIdx.x % warpSize;

    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int r = warp_id; r < R; r += gridDim.x * warp_in_block) { // schedule 1
        TVec val = init_val; // compute at schedule 1
        for (int c = lane_id ; c < C / vec_len; c += warpSize) { // schedule 2
            TVec element = reinterpret_cast<TVec *>(operand + r * C)[c]; // read element
            val = vector_reducer(val, element); // compute at schedule 2
        }

        TVec ret = warp_custom_reduce<T, ScalarReducer, 32, vec_len>(val); // compute at schedule 1

        if (lane_id == 0) {
            result[r] = to_scalar<T, vec_len, ScalarReducer>(ret); // compute as schedule 1
        }
    }
}

#include <unordered_map>

template<typename T>
void profile(int R, int C, std::string reduce_type) {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr int len_vec = 128 / cutlass::sizeof_bits<T>::value;
    using TVec = cutlass::AlignedArray<T, len_vec>;
    // using TAlignedVec = cutlass::AlignedArray<T, len_vec>;
    // using TVec = float4;

    T *operand;
    T *result;
    // checkCudaErrors(cudaMallocAsync((void **)&operand, R * C * sizeof(T), stream));
    // checkCudaErrors(cudaMallocAsync((void **)&result, R * sizeof(T), stream));
    checkCudaErrors(cudaMalloc((void **)&operand, R * C * sizeof(T)));
    checkCudaErrors(cudaMalloc((void **)&result, R * sizeof(T)));
    checkCudaErrors(cudaStreamSynchronize(stream));

    if (reduce_type == "warp") {
        reduce_warp<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, sizeof(typename cub::WarpReduce<T, 32, __CUDA_ARCH_GLOBAL__>::TempStorage) * 4, stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "warp_N") {
        reduce_warp_ILP<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, sizeof(typename cub::WarpReduce<T, 32, __CUDA_ARCH_GLOBAL__>::TempStorage) * 4, stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "warp_cub") {
        reduce_warp_cub<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, 0, stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "block") {
        reduce_block<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, 32 * sizeof(T), stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "cub1") {
        reduce_block_cub_RAKING<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, sizeof(typename cub::BlockReduce<T, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, __CUDA_ARCH_GLOBAL__>::TempStorage), stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "cub2") {
        reduce_block_cub_RAKING_COMMUTATIVE_ONLY<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, sizeof(typename cub::BlockReduce<T, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, __CUDA_ARCH_GLOBAL__>::TempStorage), stream>>>(operand, result, 0, R, C);
    }

    if (reduce_type == "cub3") {
        reduce_block_cub_WARP_REDUCTIONS<T, TVec, cutlass::plus<T>, cutlass::plus<TVec>><<<108*4, 128, sizeof(typename cub::BlockReduce<T, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, __CUDA_ARCH_GLOBAL__>::TempStorage), stream>>>(operand, result, 0, R, C);
    }

    checkCudaErrors(cudaStreamSynchronize(stream));
    // checkCudaErrors(cudaFreeAsync(operand, stream));
    // checkCudaErrors(cudaFreeAsync(result, stream));
    checkCudaErrors(cudaFree(operand));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
}

template<typename T, typename TVec>
__global__
void copy(T *src, T *dst, int n) {
    constexpr int vec_len = sizeof(TVec) / sizeof(T);

    for (int idx = vec_len * (threadIdx.x + blockIdx.x * blockDim.x); idx < n; idx += blockDim.x * gridDim.x * vec_len) {
        TVec vec = *reinterpret_cast<TVec *>(src + idx);

        // vec.x = vec.x + vec.y;
        // vec.y = vec.y + vec.z;
        // vec.z = vec.z + vec.w;
        // vec.w = vec.w + vec.x;
        vec[0] = vec[0] + vec[1];
        vec[1] = vec[1] + vec[2];
        vec[2] = vec[2] + vec[3];
        vec[3] = vec[3] + vec[0];

        *reinterpret_cast<TVec *>(dst + idx) = vec;
    }
}

void test_alignment() {
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int n = 10000000;
    float *src, *dst;
    checkCudaErrors(cudaMalloc((void **)&src, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&dst, n * sizeof(float)));

    copy<float, cutlass::AlignedArray<float, 4>><<<108, 128, 0, stream>>>(src, dst, n);
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaFree(src));
    checkCudaErrors(cudaFree(dst));

    checkCudaErrors(cudaStreamDestroy(stream));
}

__global__
void test_kernel() {
    cutlass::AlignedArray<float, 4> a1, a2, a3, a4;
    float b1, b2, b3, b4, b5;

    auto val = to_scalar<float, 4, cutlass::plus<float>>(a1, a2, a3, a4);
    auto val2 = to_scalar<float, cutlass::plus<float>>(b1, b2, b3, b4, b5);
}


int main(int argc, char const *argv[])
{
    std::cout << sizeof(cub::BlockReduce<float, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, 800>::TempStorage) << std::endl;
    std::cout << sizeof(cub::BlockReduce<float, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, 800>::TempStorage) << std::endl;
    std::cout << sizeof(cub::BlockReduce<float, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 800>::TempStorage) << std::endl;

    std::cout << sizeof(cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, 1, 1, 800>::TempStorage) << std::endl;
    std::cout << sizeof(cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 1, 1, 800>::TempStorage) << std::endl;
    std::cout << sizeof(cub::BlockReduce<half, 128, cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 800>::TempStorage) << std::endl;
    return 0;

    // test_alignment();
    profile<float>(10000, 512, "warp");


    profile<float>(108*4, 5120, "warp");
    profile<float>(108*4, 5120, "warp_cub");
    profile<float>(108*4, 5120, "block");
    profile<float>(108*4, 5120, "cub1");
    profile<float>(108*4, 5120, "cub2");
    profile<float>(108*4, 5120, "cub3");


    profile<float>(1000, 512, "warp");
    profile<float>(1000, 512, "warp_cub");
    profile<float>(1000, 512, "block");
    profile<float>(1000, 512, "cub1");
    profile<float>(1000, 512, "cub2");
    profile<float>(1000, 512, "cub3");

    profile<float>(10000, 512, "warp");
    profile<float>(10000, 512, "warp_cub");
    profile<float>(10000, 512, "block");
    profile<float>(10000, 512, "cub1");
    profile<float>(10000, 512, "cub2");
    profile<float>(10000, 512, "cub3");

    profile<float>(10000, 4096, "warp");
    profile<float>(10000, 4096, "warp_cub");
    profile<float>(10000, 4096, "block");
    profile<float>(10000, 4096, "cub1");
    profile<float>(10000, 4096, "cub2");
    profile<float>(10000, 4096, "cub3");

    profile<float>(100000, 4096, "warp");
    profile<float>(100000, 4096, "warp_cub");
    profile<float>(100000, 4096, "block");
    profile<float>(100000, 4096, "cub1");
    profile<float>(100000, 4096, "cub2");
    profile<float>(100000, 4096, "cub3");

    return 0;
}
