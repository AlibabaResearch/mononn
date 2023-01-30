#pragma once

#include <string>

#include "mononn_engine/core/gpu/cutlass/gemm_coord.h"
#include "mononn_engine/core/gpu/cutlass/gemm_universal_mode.h"
namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct GemmUniversalArgument {
  GemmUniversalMode mode;
  GemmCoord problem_size;
  std::string batch_count;
  std::string alpha, beta;
  std::string ptr_A;
  std::string ptr_B;
  std::string ptr_C;
  std::string ptr_D;
  std::string batch_stride_A;
  std::string batch_stride_B;
  std::string batch_stride_C;
  std::string batch_stride_D;
  std::string stride_a;
  std::string stride_b;
  std::string stride_c;
  std::string stride_d;

  std::string define_variable(std::string gemm_kernel,
                              std::string var_name) const;
};

struct GemmWithLoopFusionArgument {};

struct GemmWithInputFusionArgument {};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine