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

#include <memory>

#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/op.h"

namespace mononn_engine {
namespace codegen {
class HostCodegen {
 public:
  using Op = mononn_engine::core::op::Op;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using Graph = mononn_engine::core::graph::Graph;

  static std::string generate(std::shared_ptr<CUDAContext> cuda_context,
                              Graph* graph, const std::string& kernel_name);

 private:
  static std::string generate_stream(std::shared_ptr<CUDAContext> cuda_context,
                                     Graph* graph);
  static std::string generate_memory_allocation(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph);
  static std::string generate_memory_initialization(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph);
  static std::string generate_parameter_declaration(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph);
  static std::string generate_parameter_initialization(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph);
  static std::string generate_kernel_invocation(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph,
      const std::string& kernel_name);
  static std::string generate_print_output(
      std::shared_ptr<CUDAContext> cuda_context, Graph* graph);
  static std::string generate_stream_synchronize(std::string stream);
};
}  // namespace codegen
}  // namespace mononn_engine