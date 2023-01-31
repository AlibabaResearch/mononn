// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/core/gpu/cutlass/gemm_argument.h"

#include <sstream>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string GemmUniversalArgument::define_variable(std::string gemm_kernel,
                                                   std::string var_name) const {
  std::stringstream ss;
  ss << "typename " << gemm_kernel << "::Arguments " << var_name << "{\n";
  ss << this->mode.to_string() << ",\n";
  ss << this->problem_size.to_string() << ",\n";
  ss << this->batch_count << ",\n";
  ss << "{" << this->alpha << ", " << this->beta << "},\n";
  ss << this->ptr_A << ",\n";
  ss << this->ptr_B << ",\n";
  ss << this->ptr_C << ",\n";
  ss << this->ptr_D << ",\n";
  ss << this->batch_stride_A << ",\n";
  ss << this->batch_stride_B << ",\n";
  ss << this->batch_stride_C << ",\n";
  ss << this->batch_stride_D << ",\n";
  ss << this->stride_a << ",\n";
  ss << this->stride_b << ",\n";
  ss << this->stride_c << ",\n";
  ss << this->stride_d << ",\n";

  ss << "};\n";
  return ss.str();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine