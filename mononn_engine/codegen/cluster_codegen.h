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

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace codegen {
class ClusterCodegen {
 public:
  using ClusterOp = mononn_engine::core::op::ClusterOp;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  static void setup_codegen(std::shared_ptr<const CUDAContext> cuda_context,
                            std::shared_ptr<ClusterOp> cluster_op);
  static std::string generate(std::shared_ptr<const CUDAContext> cuda_context,
                              std::shared_ptr<const ClusterOp> cluster_op);
  static std::string generate_function_declaration(
      std::shared_ptr<const CUDAContext> cuda_context,
      std::shared_ptr<const ClusterOp> cluster_op);
  static std::string generate_function_definition(
      std::shared_ptr<const CUDAContext> cuda_context,
      std::shared_ptr<const ClusterOp> cluster_op);

 private:
};
}  // namespace codegen
}  // namespace mononn_engine