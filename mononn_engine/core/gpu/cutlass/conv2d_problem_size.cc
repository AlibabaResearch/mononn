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

#include "mononn_engine/core/gpu/cutlass/conv2d_problem_size.h"

#include <sstream>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
std::string Conv2dProblemSize::define_variable(
    const std::string& var_name) const {
  std::stringstream ss;
  ss << "cutlass::conv::Conv2dProblemSize " << var_name << "("
     << "\n";
  ss << "{" << this->N << "," << this->H << "," << this->W << "," << this->C
     << "},"
     << "\n";
  ss << "{" << this->K << "," << this->R << "," << this->S << "," << this->C
     << "},"
     << "\n";
  ss << "{" << this->pad_h << "," << this->pad_h << "," << this->pad_w << ","
     << this->pad_w << "},"
     << "\n";
  ss << "{" << this->stride_h << "," << this->stride_w << "},"
     << "\n";
  ss << "{" << this->dilation_h << "," << this->dilation_w << "},"
     << "\n";
  ss << "{" << this->N << "," << this->P << "," << this->Q << "," << this->K
     << "},"
     << "\n";
  ss << "cutlass::conv::Mode::kCrossCorrelation,"
     << "\n";
  ss << "1"
     << "\n";
  ss << ");"
     << "\n";
  return ss.str();
}
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine