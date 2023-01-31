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

#pragma onece

#include <string>

#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/graph/clustered_graph.h"

namespace mononn_engine {
namespace codegen {
class ClusteredGraphCodegen {
 public:
  using CUDAProgram = mononn_engine::codegen::CUDAProgram;
  using CUDAContext = mononn_engine::core::context::CUDAContext;
  using ClusteredGraph = mononn_engine::core::graph::ClusteredGraph;

  static CUDAProgram generate(std::shared_ptr<CUDAContext> cuda_context,
                              std::shared_ptr<ClusteredGraph> graph);

 private:
  static void initialize_buffer_manager(std::shared_ptr<ClusteredGraph> graph);
  static void synchronization_analysis(std::shared_ptr<ClusteredGraph> graph);
};
}  // namespace codegen
}  // namespace mononn_engine