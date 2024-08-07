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
#include <string>

#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/tensor/tensor_shape.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
class LayoutSimplificationPass : public GraphPass {
 public:
  using Op = mononn_engine::core::op::Op;
  using Graph = mononn_engine::core::graph::Graph;
  using TensorSpec = mononn_engine::core::tensor::TensorSpec;

  class LayoutTransformSpec {
   public:
    LayoutTransformSpec() {}
    LayoutTransformSpec(std::string _spec) : spec(_spec) {}
    LayoutTransformSpec(const char* _spec)
        : LayoutTransformSpec(std::string(_spec)) {}

    static LayoutTransformSpec const reshape;
    static LayoutTransformSpec const tensor_permutation;
    static LayoutTransformSpec const memory_permutation;
    static LayoutTransformSpec const reshape_tensor_permutation;
    static LayoutTransformSpec const reshape_memory_permutation;
    static LayoutTransformSpec const tensor_permutation_reshape;
    static LayoutTransformSpec const memory_permutation_reshape;

    std::vector<int> tensor_perm_spec;
    std::vector<int> memory_perm_spec;

    std::string to_string() const;

   private:
    std::string spec;
  };

  LayoutSimplificationPass() {}
  std::string name() const override;

  bool run(Graph* graph, std::shared_ptr<CUDAContext> cuda_context) override;

  LayoutTransformSpec get_layout_transform_spec(std::shared_ptr<Op> node) const;

 private:
  std::vector<int> get_sequence_perm(std::vector<int> seq1,
                                     std::vector<int> seq2) const;
};
}  // namespace optimization
}  // namespace mononn_engine
