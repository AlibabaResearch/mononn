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
#include "mononn_engine/core/graph/graph.h"
#include "tensorflow/mononn_extra/proto/graph_specification.pb.h"

namespace mononn_engine {
namespace optimization {
using CUDAContext = mononn_engine::core::context::CUDAContext;
using Graph = mononn_engine::core::graph::Graph;
using GraphSpecification = tensorflow::mononn_extra::proto::GraphSpecification;
class OptimizationRunner {
 public:
  enum Group {
    GRPOU_PRE_IMPL_ASSIGNMENT = (1 << 0),
    GRPOU_IMPL_ASSIGNMENT = (1 << 1),
    GRPOU_IMPL_OPTIMIZATIPN = (1 << 2),
    GROUP_BUFFER_ASSIGNMENT = (1 << 3),
  };

  static void run_group_pre_impl_assignment(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context);
  static void run_group_impl_assignment(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
      const GraphSpecification* graph_specification);
  static void run_group_impl_optimization(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
      const GraphSpecification* graph_specification);
  static void run_group_buffer_assignment(
      Graph* graph, std::shared_ptr<CUDAContext> cuda_context);

 private:
};
}  // namespace optimization
}  // namespace mononn_engine