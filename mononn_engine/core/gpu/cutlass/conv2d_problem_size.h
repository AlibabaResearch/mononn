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

#pragma once

#include <string>

namespace mononn_engine {
namespace core {
namespace gpu {
namespace cutlass {
struct Conv2dProblemSize {
  std::string N, H, W, C;
  std::string K, R, S;
  std::string pad_h, pad_w;
  std::string stride_h, stride_w;
  std::string dilation_h, dilation_w;
  std::string P, Q;
  std::string mode = "cutlass::conv::Mode::kCrossCorrelation";
  std::string split_k_slices = "1";
  std::string groups = "1";

  std::string define_variable(const std::string& var_name) const;
};
}  // namespace cutlass
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine
