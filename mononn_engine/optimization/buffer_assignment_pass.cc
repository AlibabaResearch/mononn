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

#include "mononn_engine/optimization/buffer_assignment_pass.h"

#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Op = mononn_engine::core::op::Op;
using Constant = mononn_engine::core::op::Constant;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;

std::string BufferAssignmentPass::name() const {
  return PassName::BufferAssignmentPass;
}

bool BufferAssignmentPass::run(Graph* graph,
                               std::shared_ptr<CUDAContext> cuda_context) {
  bool TF_MONONN_ENABLED =
      mononn_engine::helpers::EnvVar::is_true("TF_MONONN_ENABLED");

  if (TF_MONONN_ENABLED) {
    BufferManager::set_buffer_mnager_use_tf_xla_buffer(true);
  }

  // clear previous record, if any.
  BufferManager::reset();

  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    if (node->get_type() != OpType::cluster) {
      if (node->get_type() == OpType::constant &&
          std::static_pointer_cast<Constant>(node)->is_scalar())
        continue;
      if (node->get_type() == OpType::get_tuple_element) continue;
      if (node->get_type() == OpType::global_sync) continue;

      if (node->get_type() == OpType::parameter ||
          node->get_type() == OpType::constant) {
        BufferManager::buffer_in_global(node_name);
        node->set_read_from_global_memory();
      } else {
        BufferManager::buffer_in_global(node_name);
        node->set_write_to_global_memory();
      }
    }
  }

  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    if (node->get_type() == OpType::cluster) {
      for (auto const& in_cluster_node_name :
           std::static_pointer_cast<ClusterOp>(node)
               ->get_graph()
               ->get_input_nodes()) {
        std::shared_ptr<Op> in_cluster_node =
            std::static_pointer_cast<ClusterOp>(node)->get_graph()->get_node(
                in_cluster_node_name);

        BufferManager::buffer_in_global(in_cluster_node_name);
        in_cluster_node->set_read_from_global_memory();
      }

      for (auto const& in_cluster_node_name :
           std::static_pointer_cast<ClusterOp>(node)
               ->get_graph()
               ->get_output_nodes()) {
        std::shared_ptr<Op> in_cluster_node =
            std::static_pointer_cast<ClusterOp>(node)->get_graph()->get_node(
                in_cluster_node_name);
        BufferManager::buffer_in_global(in_cluster_node_name);
        in_cluster_node->set_write_to_global_memory();
      }

      BufferManager::buffer_in_global(node->get_name());
      node->set_write_to_global_memory();
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine