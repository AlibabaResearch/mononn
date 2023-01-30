#include "mononn_engine/optimization/merge_dependent_pass.h"

#include "mononn_engine/core/graph/cluster_util.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/cluster_reduce.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/helpers/stl_helpers.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "mononn_engine/helpers/transform.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterUtil = mononn_engine::core::graph::ClusterUtil;

std::string MergeDependentPass::name() const {
  return PassName::MergeDependentPass;
}

bool MergeDependentPass::run(Graph* graph,
                             std::shared_ptr<CUDAContext> cuda_context) {
  std::vector<std::string> cluster_node_list =
      graph->get_node_list_by_type(OpType::cluster);

  for (auto const& cluster_node_name : cluster_node_list) {
    auto cluster_node_begin_node = graph->get_node(cluster_node_name);

    if (!cluster_node_begin_node->as<ClusterOp>()
             ->get_graph_ptr()
             ->get_node_list_by_type(OpType::reduce_window)
             .empty()) {
      continue;
    }

    std::vector<std::string> down_stream_nodes =
        graph->search_downstream_node_under_constrain(
            cluster_node_name,
            [&](const Op* node) -> bool {
              return node->get_type() == OpType::get_tuple_element;
            },
            [&](const Op* node,
                std::vector<std::string> const& node_in_path) -> bool {
              std::vector<std::string> merge_node_list;
              if (node->get_type() != OpType::cluster) return false;
              if (cluster_node_begin_node->as<ClusterOp>()
                      ->get_loop_shape()
                      .element_count() !=
                  node->as<ClusterOp>()->get_loop_shape().element_count()) {
                return false;
              }

              if (cluster_node_begin_node->is_cluster_reduce() &&
                  node->is_cluster_reduce() &&
                  cluster_node_begin_node->as<ClusterReduce>()
                          ->get_reduction_dimension_size() !=
                      node->as<ClusterReduce>()
                          ->get_reduction_dimension_size()) {
                return false;
              }

              if (!node->as<ClusterOp>()
                       ->get_graph_ptr()
                       ->get_node_list_by_type(OpType::reduce_window)
                       .empty()) {
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
      std::shared_ptr<ClusterOp> begin_cluster =
          std::static_pointer_cast<ClusterOp>(
              graph->get_node(down_stream_nodes.front()));
      std::shared_ptr<ClusterOp> end_cluster =
          std::static_pointer_cast<ClusterOp>(
              graph->get_node(down_stream_nodes.back()));
      std::vector<std::string> path_nodes_name =
          graph->search_nodes_between_two_nodes(begin_cluster->get_name(),
                                                end_cluster->get_name());

      //                if (begin_cluster->is_cluster_elewise() &&
      //                end_cluster->is_cluster_elewise()) {
      //                    LOG(WARNING) << "Found two sequential element-wise
      //                    cluster: " << mononn_engine::helpers::join(" ",
      //                    down_stream_nodes);
      //                }
      //
      //                if (begin_cluster->is_cluster_elewise()) {
      //                    LOG(WARNING) << "Begin cluster is elewise cluster. "
      //                    << "Begin cluster: " << begin_cluster->get_name() <<
      //                    ", end cluster: " << end_cluster->get_name();
      //                }

      std::vector<std::shared_ptr<Op>> path_nodes;
      for (auto const& node_name : path_nodes_name) {
        path_nodes.push_back(graph->get_node(node_name));
      }

      std::string new_cluster_name = begin_cluster->get_name() + "_" +
                                     this->name() + "_" +
                                     end_cluster->get_name();
      std::shared_ptr<ClusterOp> new_cluster;

      if (begin_cluster->is_cluster_elewise() &&
          end_cluster->is_cluster_elewise()) {
        LOG(FATAL) << "Found two elewise clusters: "
                   << begin_cluster->get_name() << " and "
                   << end_cluster->get_name()
                   << ", two clusters should be concatenated in "
                      "ElementWiseConcatenationPass";
        // new_cluster = ClusterUtil::merge_sequential<ClusterElewise>(
        //     new_cluster_name, begin_cluster, path_nodes, end_cluster, graph);
      } else {
        new_cluster = ClusterUtil::merge_sequential<ClusterReduce>(
            new_cluster_name, begin_cluster, path_nodes, end_cluster, graph);
      }

      std::vector<std::string> nodes_to_be_merged;
      nodes_to_be_merged.insert(nodes_to_be_merged.end(),
                                down_stream_nodes.begin(),
                                down_stream_nodes.end());
      nodes_to_be_merged.insert(nodes_to_be_merged.end(),
                                path_nodes_name.begin(), path_nodes_name.end());
      //                graph->replace_node(nodes_to_be_merged, new_cluster);
      ClusterUtil::replace_node(graph, nodes_to_be_merged, new_cluster);

      if (!begin_cluster->get_graph_ptr()
               ->get_node_list_by_type(OpType::reduce_window)
               .empty()) {
        LOG(FATAL) << begin_cluster->get_name() << " has reduce-window";
      }

      if (!end_cluster->get_graph_ptr()
               ->get_node_list_by_type(OpType::reduce_window)
               .empty()) {
        LOG(FATAL) << end_cluster->get_name() << " has reduce-window";
      }

      LOG(INFO) << "Pass: " << this->name() << ". Merge "
                << begin_cluster->get_name() << " and "
                << end_cluster->get_name() << " into " << new_cluster_name;

      // LOG(INFO) << "Pass: " << this->name() << ". Merge " <<
      // begin_cluster->get_name() << " and " << end_cluster->get_name() << "
      // into " << new_cluster_name; LOG(DEBUG) << "down_stream_nodes: " <<
      // mononn_engine::helpers::join(" ", down_stream_nodes); LOG(DEBUG) <<
      // "path_nodes_name: " << mononn_engine::helpers::join(" ",
      // path_nodes_name); LOG(DEBUG) << "nodes_to_be_merged: " <<
      // mononn_engine::helpers::join(" ", nodes_to_be_merged);

      return true;
    }
  }

  return false;
}
}  // namespace optimization
}  // namespace mononn_engine