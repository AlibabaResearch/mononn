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

#include "mononn_engine/optimization/smem_prefetch_pass.h"

#include "mononn_engine/core/gpu/smem_manager.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/op_annotation/auxiliary_impl_type.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/smem_prefetch_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Tensor = mononn_engine::core::tensor::Tensor;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using AuxiliaryImplType = mononn_engine::core::op_annotation::AuxiliaryImplType;
using SmemManager = mononn_engine::core::gpu::SmemManager;
using Parameter = mononn_engine::core::op::Parameter;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using Dtype = mononn_engine::core::tensor::Dtype;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using SmemPrefetchImpl = mononn_engine::core::op_impl::SmemPrefetchImpl;
using Loop = mononn_engine::core::schedule::Loop;
using Op = mononn_engine::core::op::Op;

std::string SmemPrefetchPass::name() const {
  return PassName::SmemPrefetchPass;
}

bool sub_cluster_is_valid_for_smem_prefetch(ClusterOp* cluster_node,
                                            std::string sub_cluster_tag) {
  for (auto const& node_name : cluster_node->get_graph()->get_node_list()) {
    auto node = cluster_node->get_graph()->get_node(node_name);

    if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag)
      continue;

    if (node->get_type() == OpType::gather ||
        node->get_type() ==
            OpType::dynamic_slice ||  // Dynamic slice cannot be vectorized,
                                      // thus is unlikely profitable in
                                      // prefetch.
        node->get_type() == OpType::dynamic_update_slice ||
        node->get_type() == OpType::reduce_window ||
        node->get_type() == OpType::pad)
      return false;
  }

  for (auto const& node_name : cluster_node->get_graph()->get_node_list()) {
    auto node = cluster_node->get_graph()->get_node(node_name);

    if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag)
      continue;

    if (node->get_type() == OpType::parameter) {
      if (node->has_attribute(OpAttribute::on_chip_transfer_from_node))
        continue;

      if (node->has_attribute(OpAttribute::is_parameter_streaming_access)) {
        if (node->get_output_spec(0).get_dtype().is_vectorized()) {
          return true;
        }
      }
    }
  }

  return false;
}

void do_sub_cluster_smem_prefetch(ClusterOp* cluster_node,
                                  std::string sub_cluster_tag,
                                  SmemManager* smem_manager,
                                  std::shared_ptr<CUDAContext>& cuda_context) {
  const int async_pipeline_total_stage_count = 2;
  Loop inner_most_loop = cluster_node->get_schedule().get_inner_loop();

  cluster_node->set_attribute(OpAttribute::async_pipeline_total_stage_count,
                              std::to_string(async_pipeline_total_stage_count));

  std::vector<std::string> nodes_128bits_aligned;
  std::vector<std::string> nodes_64bits_aligned;
  std::vector<std::string> nodes_32bits_aligned;

  // int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;

  for (auto const node_name :
       cluster_node->get_graph()->traverse_in_topology_order()) {
    auto node = cluster_node->get_graph()->get_node(node_name);

    if (node->get_type() != OpType::parameter) {
      continue;
    }
    if (node->has_attribute(OpAttribute::on_chip_transfer_from_node)) continue;
    if (!node->has_attribute(OpAttribute::is_parameter_streaming_access)) {
      continue;
    }
    if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag) {
      continue;
    }
    auto type = node->get_output_spec(0).get_dtype();
    if (!type.is_vectorized()) {
      continue;
    }

    if (type.size_in_bits() == 128) {
      nodes_128bits_aligned.push_back(node_name);
    } else if (type.size_in_bits() == 64) {
      nodes_64bits_aligned.push_back(node_name);
    } else if (type.size_in_bits() == 32) {
      nodes_32bits_aligned.push_back(node_name);
    } else if (type.size_in_bits() > 128) {
      LOG(FATAL) << "Too large memory alignment of node " << node_name << " :"
                 << type.size_in_bits() << " bits";
    }
  }

  auto get_smem_base_buf_size_in_bytes = [&](Op* node) -> int {
    int inner_loop_elem_count =
        inner_most_loop.get_loop_shape().element_count();
    int highest_dim_elem_count = node->get_output_spec(0).get_shape(-1);

    // For cluster constaining slice node we should use inner_loop_elem_count
    // (which is smaller). For cluster containing pad/concatenate we should use
    // highest_dim_elem_count (which is smaller).
    // TODO any special for cluster containing both slice and pad/concat?
    int elem_count = std::min(inner_loop_elem_count, highest_dim_elem_count);

    int size_in_bytes =
        node->get_output_spec(0).get_dtype().size_in_bytes() * elem_count;
    return size_in_bytes;
  };

  auto get_smem_buf_size_in_bytes = [&](Op* node) -> int {
    int size_in_bytes = get_smem_base_buf_size_in_bytes(node);

    // Warp locality require warp_per_block times smem as each warp works
    // individually.
    if (cluster_node->get_schedule().get_locality_tier() == LocalityTier::kT1) {
      size_in_bytes *= cuda_context->cuda_runtime_context.block_dim.XYZ() /
                       32;  // multiply by warp number per block.
    }

    size_in_bytes *= async_pipeline_total_stage_count;

    return size_in_bytes;
  };

  auto buffer_in_smem_if_possible = [&](const std::string& node_name) -> bool {
    auto node = cluster_node->get_graph()->get_node(node_name);

    if (node->get_symbolic_index().size() > 1) {  // traced by multiple nodes.
      std::vector<size_t> buffer_size_list_in_bytes;
      for (auto const& symbolic_index : node->get_symbolic_index()) {
        std::string reuse_node_name =
            node_name + "_reuse_" + symbolic_index.traced_by;

        std::string smem_buffer_name = reuse_node_name + "_smem_buf";
        int smem_buf_size_in_bytes = get_smem_buf_size_in_bytes(node.get());

        buffer_size_list_in_bytes.push_back((size_t)smem_buf_size_in_bytes);
      }

      if (smem_manager->can_claim_buffer(buffer_size_list_in_bytes)) {
        for (auto const& symbolic_index : node->get_symbolic_index()) {
          std::string reuse_node_name =
              node_name + "_reuse_" + symbolic_index.traced_by;

          std::string smem_buffer_name = reuse_node_name + "_smem_buf";

          int smem_buf_size_in_bytes = get_smem_buf_size_in_bytes(node.get());

          smem_manager->claim_smem_buffer(reuse_node_name, smem_buffer_name,
                                          smem_buf_size_in_bytes);
          smem_manager->record_base_buffer_size(
              smem_buffer_name, get_smem_base_buf_size_in_bytes(node.get()));
        }

        node->set_attribute(OpAttribute::async_pipeline_total_stage_count,
                            std::to_string(async_pipeline_total_stage_count));
        node->set_attribute(OpAttribute::is_parameter_async_prefetched, "true");

        SmemPrefetchImpl::InputSpec input_spec;
        input_spec.tier = cluster_node->get_schedule().get_locality_tier();
        Tensor output(node_name, node->get_output_spec(0));
        auto impl = SmemPrefetchImpl::get_available_implementations(
            cuda_context, input_spec, output)[0];
        impl->set_hlo_text(node->get_hlo_text());
        node->set_implementation(impl);

        return true;
      }
    } else {
      int smem_buf_size_in_bytes = get_smem_buf_size_in_bytes(node.get());

      if (smem_manager->can_claim_buffer(smem_buf_size_in_bytes)) {
        std::string smem_buffer_name = node->get_name() + "_smem_buf";
        node->set_attribute(OpAttribute::async_pipeline_total_stage_count,
                            std::to_string(async_pipeline_total_stage_count));
        node->set_attribute(OpAttribute::is_parameter_async_prefetched, "true");

        smem_manager->claim_smem_buffer(node_name, smem_buffer_name,
                                        smem_buf_size_in_bytes);
        smem_manager->record_base_buffer_size(
            smem_buffer_name, get_smem_base_buf_size_in_bytes(node.get()));

        SmemPrefetchImpl::InputSpec input_spec;
        input_spec.tier = cluster_node->get_schedule().get_locality_tier();
        Tensor output(node_name, node->get_output_spec(0));
        auto impl = SmemPrefetchImpl::get_available_implementations(
            cuda_context, input_spec, output)[0];
        impl->set_hlo_text(node->get_hlo_text());
        node->set_implementation(impl);

        return true;
      }
    }

    return false;
  };

  for (auto const& node_name : nodes_128bits_aligned) {
    buffer_in_smem_if_possible(node_name);
  }

  for (auto const& node_name : nodes_64bits_aligned) {
    buffer_in_smem_if_possible(node_name);
  }

  for (auto const& node_name : nodes_32bits_aligned) {
    buffer_in_smem_if_possible(node_name);
  }
}

void do_cluster_smem_prefetch(ClusterOp* cluster_node,
                              std::shared_ptr<CUDAContext>& cuda_context) {
  SmemManager* smem_manager = cluster_node->get_smem_manager();
  bool cluster_can_prefetch = false;

  if (cluster_node->get_name() ==
      "input_fusion_reduce_43_MD_input_fusion_reduce_21_MD_fusion_105") {
    LOG(INFO) << "Cluster " << cluster_node->get_name()
              << " cannot be prefetched.";
    return;
  }

  if (cluster_node->get_cluster_type() == ClusterType::Reduce) {
    LOG_ONCE(WARNING, __log_once,
             "Only reduce cluster is supported at this stage");

    for (auto const& sub_cluster_tag :
         cluster_node->get_sub_cluster_tag_order()) {
      if (sub_cluster_is_valid_for_smem_prefetch(cluster_node,
                                                 sub_cluster_tag)) {
        LOG(INFO) << "Cluster: " << cluster_node->get_name()
                  << " sub cluster: " << sub_cluster_tag
                  << " is valid for asynchronous smem prefetch";
        do_sub_cluster_smem_prefetch(cluster_node, sub_cluster_tag,
                                     smem_manager, cuda_context);
        cluster_can_prefetch = true;
      }
    }
  }

  if (!cluster_can_prefetch) {
    LOG(INFO) << "Cluster " << cluster_node->get_name()
              << " cannot be prefetched";
  }
}

bool SmemPrefetchPass::run(Graph* graph,
                           std::shared_ptr<CUDAContext> cuda_context) {
  if (!mononn_engine::core::gpu::cutlass::Arch::newer_or_equal(
          cuda_context->cuda_device_context.get_cutlass_arch_tag(),
          mononn_engine::core::gpu::cutlass::Arch::Sm80)) {
    LOG(INFO)
        << "CUDA architecture: " << cuda_context->cuda_device_context.cuda_arch
        << " do not support LDGSTS instruction, skipping optimization pass "
        << this->name();
    return true;
  }

  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();
    do_cluster_smem_prefetch(cluster_node, cuda_context);
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine
