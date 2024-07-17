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

#include "mononn_engine/optimization/vectorization_pass.h"

#include <algorithm>
#include <set>

#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/concatenate.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/gather.h"
#include "mononn_engine/core/op/iota.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/pad.h"
#include "mononn_engine/core/op/slice.h"
#include "mononn_engine/core/op/transpose.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/schedule/loop.h"
#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using Schedule = mononn_engine::core::schedule::Schedule;
using Op = mononn_engine::core::op::Op;
using Concatenate = mononn_engine::core::op::Concatenate;
using Pad = mononn_engine::core::op::Pad;
using Slice = mononn_engine::core::op::Slice;
using Gather = mononn_engine::core::op::Gather;
using Iota = mononn_engine::core::op::Iota;
using Transpose = mononn_engine::core::op::Transpose;
using Broadcast = mononn_engine::core::op::Broadcast;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using Loop = mononn_engine::core::schedule::Loop;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using Constant = mononn_engine::core::op::Constant;

std::string VectorizationPass::name() const {
  return PassName::VectorizationPass;
}

struct NodeTraversalContext {
  bool broadcast_on_highest_dimension =
      false;  // Scalar broadcast is also considered broadcast on highest
              // dimension.
  bool scalar_broadcast = false;
  bool is_reduce_init_value = false;
  bool is_padding_value = false;
};

// Node of interest: pad slice concat gather iota transpose broadcast
bool node_can_vectorized_as(Op* node, int vec_len,
                            NodeTraversalContext& context,
                            std::set<std::string>& visit,
                            std::string sub_cluster_tag) {
  // For node does not belong to current sub cluster.
  if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag)
    return true;

  if (visit.count(node->get_name())) return true;
  visit.insert(node->get_name());

  /////////////////// maintain context ///////////////////

  if (node->get_type() == OpType::broadcast) {
    std::vector<int> dims = node->as<Broadcast>()->get_dimensions();

    // Broadcast dim do not contain highest dimension
    if (dims.empty() /*Scalar broadcast*/ ||
        std::find(dims.begin(), dims.end(),
                  node->get_output_spec(0).rank() - 1) ==
            dims.end() /*Highest dim not exists in origin operand*/) {
      context.broadcast_on_highest_dimension = true;

      if (dims.empty()) {
        context.scalar_broadcast = true;
      }
    }
  }

  /////////////////// Traverse ///////////////////

  for (int operand_id = 0; operand_id < node->get_operand_count();
       ++operand_id) {
    NodeTraversalContext next_context = context;

    if ((node->get_type() == OpType::reduce ||
         node->get_type() == OpType::reduce_window) &&
        operand_id >= node->get_operand_count() / 2) {
      next_context.is_reduce_init_value = true;
    }

    if (node->get_type() == OpType::pad && operand_id == 1) {
      next_context.is_padding_value = true;
    }

    if (!node_can_vectorized_as(node->get_operand(operand_id).get(), vec_len,
                                next_context, visit, sub_cluster_tag))
      return false;
  }

  /////////////////// check if can be vectorized. ///////////////////

  if (node->get_type() == OpType::parameter ||
      node->get_type() == OpType::constant) {
    // e.g. [128,128] <- [128, 1]
    if (context.broadcast_on_highest_dimension) return true;

    if (node->get_type() == OpType::constant &&
        (context.is_reduce_init_value || context.is_padding_value)) {
      return true;
    }

    int dim_continuous = node->get_output_spec(0).get_shape(-1);

    if (dim_continuous % vec_len != 0) {
      // LOG(DEBUG) << "Return false shape not valid " << node->get_name() << "
      // " << dim_continuous << " " << vec_len <<
      // node->get_output_spec(0).to_string();
      return false;
    }

    // Widest memory access instruction in NVIDIA GPU 128 bit.
    int vector_size_in_bits = node->get_output_spec(0)
                                  .get_dtype()
                                  .get_primitive_type()
                                  .size_in_bits() *
                              vec_len;
    if (vector_size_in_bits > 128) {
      // LOG(DEBUG) << "Return false too wide access " << vector_size_in_bits <<
      // " " << node->get_name();
      return false;
    }

    for (int valid_vector_size_in_bits = 128; valid_vector_size_in_bits >= 8;
         valid_vector_size_in_bits >>= 1) {
      if (vector_size_in_bits == valid_vector_size_in_bits) return true;
    }

    // LOG(DEBUG) << "Return false size in bit not valid " <<
    // vector_size_in_bits << " " << node->get_name();

    return false;
  }

  // if (node->get_type() == OpType::constant) {
  //     if (context.is_reduce_init_value) return true;
  //     int dim_continuous = node->get_output_spec(0).get_shape(-1);

  //     if (dim_continuous % vec_len != 0) {
  //         LOG(DEBUG) << "Constant not valid " <<
  //         node->get_output_spec(0).to_string() << " " << vec_len;
  //     }

  //     return dim_continuous % vec_len == 0;
  // }

  if (node->get_type() == OpType::transpose) {
    std::vector<int> permute = node->as<Transpose>()->get_permute();

    // transpose contain highest dimension.
    if (permute.back() != (int)permute.size() - 1) {
      // LOG(DEBUG) << "Return false Transpose check failed " <<
      // node->get_name();
      return false;
    }

    return true;
  }

  if (node->get_type() == OpType::broadcast) {
    return true;
  }

  if (node->get_type() == OpType::pad) {
    auto pad_high = node->as<Pad>()->get_padding_high();
    auto pad_low = node->as<Pad>()->get_padding_low();

    // only care about highest dimension.
    int highest_dimension = node->get_output_spec(0).get_shape(-1);
    int highest_pad_high = pad_high.back();
    int highest_pad_low = pad_low.back();

    if (highest_pad_high % vec_len != 0) {
      return false;
    }
    if (highest_pad_low % vec_len != 0) {
      return false;
    }

    if ((highest_dimension - highest_pad_high - highest_pad_low) % vec_len !=
        0) {
      return false;
    }

    return true;
  }

  if (node->get_type() == OpType::slice) {
    auto slice_starts = node->as<Slice>()->get_slice_starts();
    auto slice_limits = node->as<Slice>()->get_slice_limits();

    // this is the highest dimension after slice
    int highest_dimension = node->get_output_spec(0).get_shape(-1);
    int highest_slice_start = slice_starts.back();
    int highest_slice_limit = slice_limits.back();

    if (highest_slice_start % vec_len != 0) {
      // LOG(DEBUG) << "Return false slice " << node->get_name();

      return false;
    }
    if (highest_slice_limit % vec_len != 0) {
      // LOG(DEBUG) << "Return false slice " << node->get_name();

      return false;
    }
    if (highest_dimension % vec_len != 0) {
      // LOG(DEBUG) << "Return false slice " << node->get_name();

      return false;
    }

    return true;
  }

  if (node->get_type() == OpType::concatenate) {
    int concat_dim = node->as<Concatenate>()->get_dimension();

    // concat can be vectorized if all of its operands can be vectoried.
    return true;
  }

  if (node->get_type() == OpType::gather) {
    // do not vectorize gather at this monent.
    return false;
  }

  if (node->get_type() == OpType::dynamic_slice ||
      node->get_type() == OpType::dynamic_update_slice) {
    return false;
  }

  if (node->get_type() == OpType::reduce_window) {
    return false;
  }

  if (node->get_type() == OpType::iota) {
    return true;
  }

  if (node->get_type() == OpType::reduce) {
    return node->get_operand(0)->get_output_spec(0).get_shape(-1) % vec_len ==
           0;
  }

  return true;
}

bool sub_cluster_can_vectorized_as(ClusterOp* cluster_node, int vec_len,
                                   std::string sub_cluster_tag) {
  // if (cluster_node->get_loop_shape().get_shape(-1) % vec_len != 0) return
  // false; if
  // (cluster_node->get_schedule().get_inner_loop().get_loop_shape().element_count()
  // % vec_len != 0) return false;

  std::vector<std::string> node_name_in_order;
  for (auto const& node_name :
       cluster_node->get_graph()->traverse_in_topology_order()) {
    auto node = cluster_node->get_graph()->get_node(node_name);
    if (node->get_attribute(OpAttribute::sub_cluster_tag) == sub_cluster_tag) {
      node_name_in_order.push_back(node_name);
    }
  }

  std::set<std::string> visit;

  for (auto node_name_it = node_name_in_order.rbegin();
       node_name_it != node_name_in_order.rend(); ++node_name_it) {
    NodeTraversalContext context;
    if (!node_can_vectorized_as(
            cluster_node->get_graph()->get_node(*node_name_it).get(), vec_len,
            context, visit, sub_cluster_tag))
      return false;
  }

  return true;
}

void do_node_vectorize(Op* node, int vec_len, NodeTraversalContext& context,
                       std::set<std::string>& visit,
                       std::string sub_cluster_tag) {
  if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag) {
    return;
  }

  std::string node_name = node->get_name();

  if (visit.count(node_name)) {
    return;
  }
  visit.insert(node_name);

  /// Maintain state ///

  bool first_semi_vectorized_broadcast = false;

  if (node->get_type() == OpType::broadcast) {
    std::vector<int> dims = node->as<Broadcast>()->get_dimensions();

    // Broadcast dim do not contain highest dimension
    if (dims.empty() /*Scalar broadcast*/ ||
        std::find(dims.begin(), dims.end(),
                  node->get_output_spec(0).rank() - 1) == dims.end()) {
      if (context.broadcast_on_highest_dimension == false) {
        first_semi_vectorized_broadcast = true;
      }

      context.broadcast_on_highest_dimension = true;

      if (dims.empty()) {
        context.scalar_broadcast = true;
      }

      node->set_attribute(OpAttribute::is_broadcast_semi_vectorized, "true");
    }
  }

  /// Traversal ///
  for (int operand_id = 0; operand_id < node->get_operand_count();
       ++operand_id) {
    auto operand = node->get_operand(operand_id);

    NodeTraversalContext next_context = context;
    do_node_vectorize(operand.get(), vec_len, next_context, visit,
                      sub_cluster_tag);
  }

  /// do vectorize ///
  if (node->get_type() == OpType::constant) {
    if (node->as<Constant>()->is_scalar()) return;
  }

  if (node->get_type() == OpType::reduce) return;

  if (context.broadcast_on_highest_dimension &&
      !first_semi_vectorized_broadcast) {
    node->set_attribute(OpAttribute::is_node_stop_vectorized, "true");
    return;
  }

  if (node->get_type() == OpType::slice) {
    std::vector<int> slice_starts = node->as<Slice>()->get_slice_starts();
    std::vector<int> slice_limits = node->as<Slice>()->get_slice_limits();

    slice_starts.back() = slice_starts.back() / vec_len;
    slice_limits.back() = slice_limits.back() / vec_len;

    node->as<Slice>()->set_slice_starts(slice_starts);
    node->as<Slice>()->set_slice_limits(slice_limits);
  }

  if (node->get_type() == OpType::pad) {
    auto pad_high = node->as<Pad>()->get_padding_high();
    auto pad_low = node->as<Pad>()->get_padding_low();

    pad_high.back() = pad_high.back() / vec_len;
    pad_low.back() = pad_low.back() / vec_len;

    node->as<Pad>()->set_padding_high(pad_high);
    node->as<Pad>()->set_padding_low(pad_low);
  }

  std::vector<TensorSpec> output_specs = node->get_output_specs();

  // check if node already vectorized.
  if (std::any_of(output_specs.begin(), output_specs.end(),
                  [](const TensorSpec& spec) -> bool {
                    return spec.get_dtype().is_vectorized();
                  })) {
    return;
  }

  for (auto& spec : output_specs) {
    if (spec.get_shape(-1) % vec_len != 0 &&
        node->get_type() != OpType::bitcast) {
      LOG(FATAL) << "Node " << node_name << node->get_output_spec(0).to_string()
                 << " cannot be vectorized by " << vec_len;
    }

    // Bitcast may disturb vectorizaiotn. for example [16,1] <- bitcast [16]
    // Result in vectorizaiton failure.
    if (node->get_type() == OpType::bitcast && spec.get_shape(-1) == 1) {
      if (spec.get_shape().rank() > 1 && spec.get_shape(-2) % vec_len == 0 &&
          spec.get_shape(-2) ==
              node->get_operand(0)->get_output_spec(0).get_shape(-1) *
                  node->get_operand(0)
                      ->get_output_spec(0)
                      .get_dtype()
                      .get_elements_per_access()) {
        auto shape = spec.get_shape();
        auto layout = spec.get_layout();
        auto dtype = spec.get_dtype();

        shape.set_shape(-2, shape.get_shape(-2) / vec_len);
        dtype = dtype.vectorize(vec_len);

        spec = TensorSpec(dtype, shape, layout);
      } else {
        LOG(FATAL) << "Node " << node_name
                   << node->get_output_spec(0).to_string()
                   << " cannot be vectorized by " << vec_len
                   << ". Type: " << node->get_type().to_string()
                   << " shape: " << spec.to_string()
                   << ", operand: " << node->get_operand(0)->get_name()
                   << ", shape: "
                   << node->get_operand(0)->get_output_spec(0).to_string();
      }
    } else {
      spec = spec.vectorize(vec_len);
    }
  }

  node->set_output_specs(output_specs);
}

// currently we only use the same vectorization length for all sub-clusters in
// the same cluster.
void do_vectorize(ClusterOp* cluster_node, int vec_len) {
  NodeTraversalContext context;

  for (auto const& sub_cluster_tag :
       cluster_node->get_sub_cluster_tag_order()) {
    std::vector<std::string> node_name_in_order;
    for (auto const& node_name :
         cluster_node->get_graph()->traverse_in_topology_order()) {
      auto node = cluster_node->get_graph()->get_node(node_name);
      if (node->get_attribute(OpAttribute::sub_cluster_tag) ==
          sub_cluster_tag) {
        node_name_in_order.push_back(node_name);
      }
    }

    std::set<std::string> visit;

    for (auto node_name_it = node_name_in_order.rbegin();
         node_name_it != node_name_in_order.rend(); ++node_name_it) {
      auto node = cluster_node->get_graph()->get_node(*node_name_it);
      context.broadcast_on_highest_dimension = false;
      do_node_vectorize(node.get(), vec_len, context, visit, sub_cluster_tag);
    }
  }
}

void vectorize_cluster(ClusterOp* cluster_node) {
  // LOG(DEBUG) << "=======Try vectorize " << cluster_node->get_name();
  for (int vec_len = 16; vec_len > 1; vec_len >>= 1) {
    // LOG(DEBUG) << "=====Try vectorize " << cluster_node->get_name() << " with
    // len " << vec_len;
    bool can_vectorize = true;
    for (auto const& sub_cluster_tag :
         cluster_node->get_sub_cluster_tag_order()) {
      if (!sub_cluster_can_vectorized_as(cluster_node, vec_len,
                                         sub_cluster_tag)) {
        can_vectorize = false;
        break;
      }
    }

    if (can_vectorize) {
      LOG(INFO) << "Vectorize cluster node " << cluster_node->get_name()
                << " to " << vec_len << "...";
      do_vectorize(cluster_node, vec_len);
      return;
    }
  }

  LOG(INFO) << "Cluster node " << cluster_node->get_name()
            << " cannot be vectorized";
}

bool VectorizationPass::run(Graph* graph,
                            std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    // if (cluster_node_name != "fusion_256_MD_fusion_79_MD_fusion_77")
    // continue;

    vectorize_cluster(graph->get_node(cluster_node_name)->as<ClusterOp>());
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine