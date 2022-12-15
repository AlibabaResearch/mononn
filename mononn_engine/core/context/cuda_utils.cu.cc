#include <cuda_runtime.h>
#include <iostream>
#include "mononn_engine/core/context/cuda_utils.h"

namespace mononn_engine {
namespace core {
namespace context {

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

    __global__ void mock_kernel() {}

// On A100
// TB/SM| max per block sm size | per sm reserved sm size
// 1: 166912 1024
// 2: 82944 2048
// 3: 54912 3200
// 4: 40960 4096
// 5: 32512 5376
// 6: 26880 6656

    int get_max_smem_size_per_block(int desired_block_count_per_sm, int block_size, int max_configurable_smem) {
        cudaFuncSetAttribute(mock_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_configurable_smem);
        int L = 0, R = max_configurable_smem;

        while (L <= R) {
            int M = L + (R - L) / 2;
            int numBlocks;

            checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &numBlocks,
                    mock_kernel,
                    block_size,
                    M));

            if (numBlocks < desired_block_count_per_sm) R = M - 1;
            else L = M + 1;
        }

        return R;
    }
}
}
}

