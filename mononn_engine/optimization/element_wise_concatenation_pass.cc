#include "mononn_engine/optimization/element_wise_concatenation_pass.h"

#include "mononn_engine/core/graph/cluster_util.h"
#include "mononn_engine/core/op/cluster_elewise.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpType = mononn_engine::core::op::OpType;
using ClusterUtil = mononn_engine::core::graph::ClusterUtil;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;

std::string ElementWiseConcatenationPass::name() const {
  return PassName::ElementWiseConcatenationPass;
}

bool ElementWiseConcatenationPass::run(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    auto cluster_node =
        std::static_pointer_cast<ClusterOp>(graph->get_node(cluster_node_name));

    if (cluster_node->is_cluster_elewise()) {
      std::vector<std::string> down_stream_nodes =
          graph->search_downstream_node_under_constrain(
              cluster_node_name,
              [&](const Op* node) -> bool {
                return node->get_type() == OpType::get_tuple_element;
              },
              [&](const Op* node,
                  std::vector<std::string> const& node_in_path) -> bool {
                std::vector<std::string> merge_node_list;
                if (!node->is_cluster_elewise()) return false;
                auto cluster_node_begin_node =
                    graph->get_node(cluster_node_name);
                if (cluster_node_begin_node->as<ClusterOp>()
                        ->get_loop_shape()
                        .element_count() !=
                    node->as<ClusterOp>()->get_loop_shape().element_count()) {
                  return false;
                }

                std::vector<std::string> nodes_between_two_nodes =
                    graph->search_nodes_between_two_nodes(cluster_node_name,
                                                          node->get_name());

                std::vector<std::string> get_tuple_element_between_two_nodes;
                for (auto const& node_name : nodes_between_two_nodes) {
                  if (graph->get_node(node_name)->get_type() ==
                      OpType::get_tuple_element) {
                    get_tuple_element_between_two_nodes.push_back(node_name);
                  }
                }

                merge_node_list.push_back(cluster_node_name);
                merge_node_list.push_back(node->get_name());
                merge_node_list = mononn_engine::helpers::vector_concat(
                    merge_node_list, get_tuple_element_between_two_nodes);

                return graph->remain_acyclic_after_node_merge(merge_node_list);
              });

      if (!down_stream_nodes.empty()) {
        std::shared_ptr<ClusterOp> end_cluster =
            std::static_pointer_cast<ClusterOp>(
                graph->get_node(down_stream_nodes.back()));
        std::vector<std::string> path_nodes_name =
            graph->search_nodes_between_two_nodes(cluster_node->get_name(),
                                                  end_cluster->get_name());

        std::vector<std::shared_ptr<Op>> path_nodes;
        for (auto const& node_name : path_nodes_name) {
          path_nodes.push_back(graph->get_node(node_name));
        }

        std::string new_cluster_name = cluster_node->get_name() + "_" +
                                       this->name() + "_" +
                                       end_cluster->get_name();
        std::shared_ptr<ClusterOp> new_cluster =
            ClusterUtil::merge_sequential<ClusterElewise>(
                new_cluster_name, cluster_node, path_nodes, end_cluster, graph);

        std::vector<std::string> nodes_to_be_merged;
        nodes_to_be_merged.insert(nodes_to_be_merged.end(),
                                  down_stream_nodes.begin(),
                                  down_stream_nodes.end());
        nodes_to_be_merged.insert(nodes_to_be_merged.end(),
                                  path_nodes_name.begin(),
                                  path_nodes_name.end());

        ClusterUtil::replace_node(graph, nodes_to_be_merged, new_cluster);

        LOG(INFO) << "Pass: " << this->name() << ". Concatenate "
                  << cluster_node->get_name() << " and "
                  << end_cluster->get_name() << " into " << new_cluster_name;

        return true;
      }
    }
  }

  return false;
}

}  // namespace optimization
}  // namespace mononn_engine