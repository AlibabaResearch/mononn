#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace limits {
    template<typename T>
    struct numeric_limits;

    template<>
    struct numeric_limits<float> {
        __device__ __forceinline__
        static float max() {
            return __FLT_MAX__;
        }
    };

    template<>
    struct numeric_limits<int32_t> {
        __device__ __forceinline__
        static int32_t max() {
            return __INT32_MAX__;
        }
    };

    template<>
    struct numeric_limits<half> {
        __device__ __forceinline__
        static half max() {
            return half(65504.0);
        }
    };
}

__global__
void run() {
    auto float_max = limits::numeric_limits<float>::max;
    auto int_max = limits::numeric_limits<int32_t>::max;
    auto half_max = limits::numeric_limits<half>::max;
}
int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
