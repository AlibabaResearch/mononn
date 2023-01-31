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
#include <vector>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/core/tensor/tensor_spec.h"

namespace mononn_engine {
namespace core {
namespace gpu {
class Reduction {
 public:
  using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
  using Tier = mononn_engine::core::op_annotation::LocalityTier::Tier;
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;
  using Functor = mononn_engine::core::gpu::Functor;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Tensor = mononn_engine::core::tensor::Tensor;

  Reduction() = delete;

  static std::vector<std::string> get_implementations();
  static std::string get_op_definition();
  static std::vector<FunctionInvocation> get_invocations(
      std::shared_ptr<CUDAContext> context, const Tensor& operand,
      const Tier& tier, const Functor& reducer);

 private:
};
}  // namespace gpu
}  // namespace core
}  // namespace mononn_engine