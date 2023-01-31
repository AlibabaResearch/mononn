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

#include "mononn_engine/optimization/cache_prefetch_pass.h"

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op_annotation/auxiliary_impl_type.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/cache_prefetch_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Tensor = mononn_engine::core::tensor::Tensor;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using AuxiliaryImplType = mononn_engine::core::op_annotation::AuxiliaryImplType;
using CachePrefetchImpl = mononn_engine::core::op_impl::CachePrefetchImpl;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;

std::string CachePrefetchPass::name() const {
  return PassName::CachePrefetchPass;
}

bool CachePrefetchPass::run(Graph* graph,
                            std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node =
        std::static_pointer_cast<ClusterOp>(graph->get_node(cluster_node_name));
    for (auto const& node_name :
         cluster_node->get_graph_ptr()->get_node_list()) {
      auto node = cluster_node->get_graph_ptr()->get_node(node_name);
      if (node->get_type() == OpType::parameter &&
          !node->has_attribute(OpAttribute::on_chip_transfer_from_node)) {
        CachePrefetchImpl::InputSpec input_spec;
        input_spec.operand = Tensor(node->get_name(), node->get_output_spec(0));
        auto cache_prefetch_impl =
            CachePrefetchImpl::get_available_implementations(cuda_context,
                                                             input_spec)[0];
        cache_prefetch_impl->set_hlo_text("Cache prefetch");
        node->add_auxiliary_impl(AuxiliaryImplType::cache_prefetch,
                                 cache_prefetch_impl);
        node->set_attribute(OpAttribute::is_parameter_cache_prefetched, "true");
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine