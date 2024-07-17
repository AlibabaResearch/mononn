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

#include "mononn_engine/optimization/layout_simplification_pass.h"

#include <numeric>
#include <sstream>

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using LayoutTransformSpec = LayoutSimplificationPass::LayoutTransformSpec;

std::string LayoutSimplificationPass::name() const {
  return "LayoutSimplificationPass";
}

LayoutTransformSpec const LayoutTransformSpec::reshape = "reshape";
LayoutTransformSpec const LayoutTransformSpec::tensor_permutation =
    "tensor_permutation";  // transpose: [2,3]{1,0} -> [3,2]{0,1}
LayoutTransformSpec const LayoutTransformSpec::memory_permutation =
    "memory_permutation";  // row->col major [2,3]{1,0} -> [2,3]->{0,1}
LayoutTransformSpec const LayoutTransformSpec::reshape_tensor_permutation =
    "reshape_tensor_permutation";
LayoutTransformSpec const LayoutTransformSpec::reshape_memory_permutation =
    "reshape_memory_permutation";
LayoutTransformSpec const LayoutTransformSpec::tensor_permutation_reshape =
    "tensor_permutation_reshape";
LayoutTransformSpec const LayoutTransformSpec::memory_permutation_reshape =
    "memory_permutation_reshape";

std::string LayoutSimplificationPass::LayoutTransformSpec::to_string() const {
  std::stringstream ss;
  ss << this->spec << " ";

  if (!this->tensor_perm_spec.empty())
    ss << "Tensor perm: "
       << mononn_engine::helpers::to_string(this->tensor_perm_spec);
  if (!this->memory_perm_spec.empty())
    ss << "Memory perm: "
       << mononn_engine::helpers::to_string(this->memory_perm_spec);

  return ss.str();
}

bool LayoutSimplificationPass::run(Graph* graph,
                                   std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);

    if (node->get_type() == OpType::copy ||
        node->get_type() == OpType::bitcast ||
        node->get_type() == OpType::reshape) {
      TensorSpec operand = node->get_operand(0)->get_output_spec(0);
      TensorSpec output = node->get_output_spec(0);

      LayoutTransformSpec spec = this->get_layout_transform_spec(node);
    }
  }

  return true;
}

LayoutTransformSpec LayoutSimplificationPass::get_layout_transform_spec(
    std::shared_ptr<Op> node) const {
  LayoutTransformSpec spec;

  if (node->get_operands().size() != 1)
    LOG(FATAL) << "Node " << node->get_name()
               << "should have exactly one operand, got "
               << node->get_operands().size();
  if (node->get_operand(0)->get_output_specs().size() != 1)
    LOG(FATAL) << "Operand of " << node->get_name()
               << " should have exactly one output, got "
               << node->get_operand(0)->get_output_specs().size();
  if (node->get_output_specs().size() != 1)
    LOG(FATAL) << "Node of " << node->get_name()
               << "should have exactly one output";

  TensorSpec operand = node->get_operand(0)->get_output_spec(0);
  TensorSpec output = node->get_output_spec(0);

  if (operand.get_dtype() != output.get_dtype())
    LOG(FATAL) << "Dtype not match for node " << node->get_name() << " from "
               << operand.get_dtype().to_string() << " to "
               << output.get_dtype().to_string();

  if (operand.rank() == output.rank()) {
    if (operand.get_shape() == output.get_shape() &&
        !(operand.get_layout() == output.get_layout())) {
      spec = LayoutTransformSpec::memory_permutation;
      spec.memory_perm_spec = this->get_sequence_perm(
          operand.get_layout().get_layout(), output.get_layout().get_layout());
      return spec;  // memory permutation
    }

    if (this->get_sequence_perm(operand.get_shape().get_shape(),
                                output.get_shape().get_shape()) ==
        this->get_sequence_perm(operand.get_layout().get_layout(),
                                output.get_layout().get_layout())) {
      spec = LayoutTransformSpec::tensor_permutation;
      spec.tensor_perm_spec = this->get_sequence_perm(
          operand.get_shape().get_shape(), output.get_shape().get_shape());

      return spec;  // tensor permutation
    }

    LOG(FATAL) << "Unrecognized permutation from " << operand.to_string()
               << " to " << output.to_string();
  }

  if (operand.can_reshape_to(output.get_shape().get_shape())) {
    TensorSpec intermediate_target =
        operand.reshape(output.get_shape().get_shape());

    if (intermediate_target.get_layout() == output.get_layout()) {
      spec = LayoutTransformSpec::reshape;
      return spec;  // reshape
    }

    spec = LayoutTransformSpec::reshape_memory_permutation;
    spec.memory_perm_spec =
        this->get_sequence_perm(intermediate_target.get_layout().get_layout(),
                                output.get_layout().get_layout());

    return spec;  // reshape + memory permutation
  }

  std::vector<int> intermediate_target_shape = output.get_shape().get_shape();

  std::sort(intermediate_target_shape.begin(), intermediate_target_shape.end());

  do {
    if (operand.can_reshape_to(intermediate_target_shape)) {
      TensorSpec intermediate_target =
          operand.reshape(intermediate_target_shape);

      if (this->get_sequence_perm(intermediate_target.get_shape().get_shape(),
                                  output.get_shape().get_shape()) ==
          this->get_sequence_perm(intermediate_target.get_layout().get_layout(),
                                  output.get_layout().get_layout())) {
        spec = LayoutTransformSpec::reshape_tensor_permutation;
        spec.tensor_perm_spec =
            this->get_sequence_perm(intermediate_target.get_shape().get_shape(),
                                    output.get_shape().get_shape());

        return spec;  // reshape + tensor permutation
      }
    }
  } while (std::next_permutation(intermediate_target_shape.begin(),
                                 intermediate_target_shape.end()));

  std::vector<int> intermediate_permutation(operand.rank());
  std::iota(intermediate_permutation.begin(), intermediate_permutation.end(),
            0);

  do {
    TensorSpec intermediate_target =
        operand.tensor_permutation(intermediate_permutation);
    if (intermediate_target.can_reshape_to(output.get_shape().get_shape())) {
      if (intermediate_target.reshape(output.get_shape().get_shape()) != output)
        continue;

      spec = LayoutTransformSpec::tensor_permutation_reshape;
      spec.tensor_perm_spec = intermediate_permutation;

      return spec;  // tensor permutation + reshape
    }
  } while (std::next_permutation(intermediate_permutation.begin(),
                                 intermediate_permutation.end()));

  std::iota(intermediate_permutation.begin(), intermediate_permutation.end(),
            0);

  do {
    TensorSpec intermediate_target =
        operand.memory_permutation(intermediate_permutation);
    if (intermediate_target.can_reshape_to(output.get_shape().get_shape())) {
      if (intermediate_target.reshape(output.get_shape().get_shape()) != output)
        continue;

      spec = LayoutTransformSpec::memory_permutation_reshape;
      spec.memory_perm_spec = intermediate_permutation;

      return spec;  // memory permutation + reshape
    }
  } while (std::next_permutation(intermediate_permutation.begin(),
                                 intermediate_permutation.end()));

  LOG(FATAL) << "Cannot figure out layout transformation specification from "
             << operand.to_string() << " to " << output.to_string();
}

std::vector<int> LayoutSimplificationPass::get_sequence_perm(
    std::vector<int> seq1, std::vector<int> seq2) const {
  std::vector<int> seq1_sort = seq1;
  std::vector<int> seq2_sort = seq2;

  std::sort(seq1_sort.begin(), seq1_sort.end());
  std::sort(seq2_sort.begin(), seq2_sort.end());

  if (!(seq1_sort == seq2_sort))
    LOG(FATAL) << "Two sequence do not have valid permutation.";

  std::vector<int> perm;

  for (int idx = 0; idx < (int)seq1.size(); ++idx) {
    auto it = std::find(seq1.begin(), seq1.end(), seq2[idx]);
    if (it == seq1.end()) LOG(FATAL) << "Element not in sequence";

    perm.push_back(it - seq1.begin());
  }

  return perm;
}
}  // namespace optimization
}  // namespace mononn_engine