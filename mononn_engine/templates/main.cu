#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cutlass/functional.h"
#include "cutlass/arch/memory.h"
#include "cutlass/functional_addon.h"
#include "cutlass/array.h"


template<typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;

template<typename T, int N>
using Array = cutlass::Array<T, N>;

template<typename T, int N>
struct Concat {
  template<typename ... TArgs>
  __device__ __forceinline__
  static AlignedArray<T, N> do_concat(bool &pred, AlignedArray<T, N> &val, TArgs &... args) {
    return pred ? val : Concat<T, N>::do_concat(args...);
  }

  template<typename ... TArgs>
  __device__ __forceinline__
  static Array<T, N> do_concat(bool &pred, Array<T, N> &val, TArgs &... args) {
    return pred ? val : Concat<T, N>::do_concat(args...);
  }

  __device__ __forceinline__
  static AlignedArray<T, N> do_concat(AlignedArray<T, N> &val) {
    return val;
  }

  __device__ __forceinline__
  static Array<T, N> do_concat(Array<T, N> &val) {
    return val;
  }
};


template<typename T>
struct Concat<T, 1> {
  template<typename ... TArgs>
  __device__ __forceinline__
  static T do_concat(bool &pred, T &val, TArgs &... args) {
    return pred ? val : Concat<T, 1>::do_concat(args...);
  }

  __device__ __forceinline__
  static T do_concat(T &val) {
    return val;
  }
};

__global__
void test(float *src, float *dst) {
    bool pred1 = false;
    bool pred2 = false;

    cutlass::Array<float, 4> v1, v2, v3;

    auto res = Concat<float, 4>::do_concat(pred1, v1, pred2, v2, v3);
}

int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
