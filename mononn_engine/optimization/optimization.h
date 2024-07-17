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
