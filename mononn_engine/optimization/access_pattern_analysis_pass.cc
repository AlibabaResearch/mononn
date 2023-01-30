#include "mononn_engine/optimization/access_pattern_analysis_pass.h"

#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/transpose.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using Broadcast = mononn_engine::core::op::Broadcast;
using Transpose = mononn_engine::core::op::Transpose;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpType = mononn_engine::core::op::OpType;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;

std::string AccessPatternAnalysisPass::name() const {
  return PassName::AccessPatternAnalysisPass;
}

struct NodeTraversalContext {
  bool broadcast_on_highest_dimension = false;
  bool broadcasted = false;
  bool transposed_highest_dimension = false;
};

void analyse(Op* node, NodeTraversalContext& context,
             std::set<std::string>& visit, std::string sub_cluster_tag) {
  // For node does not belong to current sub cluster.
  if (node->get_attribute(OpAttribute::sub_cluster_tag) != sub_cluster_tag)
    return;

  if (visit.count(node->get_name())) return;

  visit.insert(node->get_name());

  if (node->get_type() == OpType::broadcast) {
    context.broadcasted = true;

    std::vector<int> dims = node->as<Broadcast>()->get_dimensions();

    // Broadcast dim do not contain highest dimension
    if (!dims.empty() &&
        std::find(dims.begin(), dims.end(),
                  node->get_output_spec(0).rank() - 1) == dims.end()) {
      context.broadcast_on_highest_dimension = true;
    }
  }

  if (node->get_type() == OpType::transpose) {
    std::vector<int> permute = node->as<Transpose>()->get_permute();

    // transpose contain highest dimension.
    if (permute.back() != (int)permute.size() - 1) {
      context.transposed_highest_dimension = true;
    }
  }

  for (int operand_id = 0; operand_id < node->get_operand_count();
       ++operand_id) {
    auto operand = node->get_operand(operand_id).get();
    NodeTraversalContext next_context = context;
    analyse(operand, next_context, visit, sub_cluster_tag);
  }

  if (node->get_type() == OpType::parameter) {
    if (context.broadcasted) {
      node->set_attribute(OpAttribute::is_parameter_temporal_access, "true");
    } else if (!context.transposed_highest_dimension) {
      node->set_attribute(OpAttribute::is_parameter_streaming_access, "true");
    }
  }
}

void analyse_sub_cluster(ClusterOp* cluster_node, std::string sub_cluster_tag) {
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
    auto node = cluster_node->get_graph()->get_node(*node_name_it);
    NodeTraversalContext context;
    analyse(node.get(), context, visit, sub_cluster_tag);
  }
}

void analyse_cluster(ClusterOp* cluster_node) {
  for (auto sub_cluster_tag : cluster_node->get_sub_cluster_tag_order()) {
    analyse_sub_cluster(cluster_node, sub_cluster_tag);
  }
}

bool AccessPatternAnalysisPass::run(Graph* graph,
                                    std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();
    analyse_cluster(cluster_node);

    // Temporal buffer not larger than l1_cache_limit_in_bytes (20KB) to prevent
    // cache thrashing.
    std::vector<Op*> parameter_temporal_access_list;

    for (auto const& node_name :
         cluster_node->get_graph_ptr()->get_node_list()) {
      auto node = cluster_node->get_graph_ptr()->get_node(node_name);

      if (node->has_attribute(OpAttribute::is_parameter_temporal_access) &&
          node->get_attribute(OpAttribute::is_parameter_temporal_access) ==
              "true") {
        parameter_temporal_access_list.push_back(node.get());
      }
    }

    std::sort(parameter_temporal_access_list.begin(),
              parameter_temporal_access_list.end(),
              [](const Op* a, const Op* b) -> bool {
                return a->get_output_spec(0).size_in_bytes() <
                       b->get_output_spec(0).size_in_bytes();
              });

    int bufferred_size_in_bytes = 0;

    for (auto* node : parameter_temporal_access_list) {
      if (bufferred_size_in_bytes + node->get_output_spec(0).size_in_bytes() <=
          l1_cache_limit_in_bytes) {
        bufferred_size_in_bytes += node->get_output_spec(0).size_in_bytes();
      } else {
        node->del_attribute(OpAttribute::is_parameter_temporal_access);
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine