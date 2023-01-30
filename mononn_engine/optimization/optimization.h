#pragma once

#include "mononn_engine/codegen/cuda_program.h"

namespace mononn_engine {
namespace optimization {
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
class Optimization {
 public:
  static std::unique_ptr<CUDAProgram> optimize();
};
}  // namespace optimization
}  // namespace mononn_engine
