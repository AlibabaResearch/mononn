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

#include "mononn_engine/optimization/trace_symbolic_index_pass.h"

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;

std::string TraceSymbolicIndexPass::name() const {
  return PassName::TraceSymbolicIndexPass;
}

bool TraceSymbolicIndexPass::run(Graph* graph,
                                 std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();

    if (cluster_node->get_cluster_type() == ClusterType::Elewise ||
        cluster_node->get_cluster_type() == ClusterType::Reduce) {
      cluster_node->trace_symbolic_index();
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine