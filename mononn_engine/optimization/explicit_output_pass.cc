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

#include "mononn_engine/optimization/explicit_output_pass.h"

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/output.h"
#include "mononn_engine/core/op_annotation/auxiliary_impl_type.h"
#include "mononn_engine/core/op_impl/output_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using Output = mononn_engine::core::op::Output;
using Op = mononn_engine::core::op::Op;
using OutputImpl = mononn_engine::core::op_impl::OutputImpl;
using Tensor = mononn_engine::core::tensor::Tensor;
using AuxiliaryImplType = mononn_engine::core::op_annotation::AuxiliaryImplType;

std::string ExplicitOutputPass::name() const {
  return PassName::ExplicitOutputPass;
}

bool ExplicitOutputPass::run(Graph* graph,
                             std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    std::shared_ptr<ClusterOp> cluster_node =
        std::static_pointer_cast<ClusterOp>(graph->get_node(cluster_node_name));

    for (int idx = 0; idx < cluster_node->get_graph()->get_output_node_count();
         ++idx) {
      std::string node_name = cluster_node->get_graph()->get_output_node(idx);
      std::shared_ptr<Op> node = cluster_node->get_graph()->get_node(node_name);

      if (node->get_type() == OpType::reduce) continue;

      OutputImpl::InputSpec input_spec;
      input_spec.operand = Tensor(node_name, node->get_output_spec(0));
      std::shared_ptr<OutputImpl> output_impl =
          std::make_shared<OutputImpl>(cuda_context, input_spec);
      output_impl->set_hlo_text("// Explicit output node");
      node->add_auxiliary_impl(AuxiliaryImplType::explicit_output_node,
                               output_impl);
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine