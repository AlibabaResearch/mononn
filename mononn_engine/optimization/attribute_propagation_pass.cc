#include "mononn_engine/optimization/attribute_propagation_pass.h"

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using OpType = mononn_engine::core::op::OpType;
using Graph = mononn_engine::core::graph::Graph;

std::string AttributePropagationPass::name() const {
  return PassName::AttributePropagationPass;
}

void propagate_parameter_attributes(Graph* graph) {
  std::vector<std::string> attribute_propagate_list = {
      OpAttribute::is_broadcast_semi_vectorized,
      OpAttribute::is_node_stop_vectorized,
      OpAttribute::is_parameter_temporal_access,
      OpAttribute::is_parameter_streaming_access,
      OpAttribute::async_pipeline_total_stage_count};

  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();
    for (auto const& node_name :
         cluster_node->get_graph()->get_node_list_by_type(OpType::parameter)) {
      auto node = cluster_node->get_graph()->get_node(node_name);

      for (auto const& attr : attribute_propagate_list) {
        if (node->has_attribute(attr)) {
          node->get_implementation()->set_attribute(attr,
                                                    node->get_attribute(attr));
        }
      }
    }
  }
}

bool AttributePropagationPass::run(Graph* graph,
                                   std::shared_ptr<CUDAContext> cuda_context) {
  propagate_parameter_attributes(graph);

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine