#include "mononn_engine/optimization/merge_independent_pass.h"

#include "mononn_engine/core/graph/cluster_util.h"
#include "mononn_engine/core/op/cluster_elewise.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/cluster_reduce.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/optimization/common.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace optimization {
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using ClusterUtil = mononn_engine::core::graph::ClusterUtil;
using Op = mononn_engine::core::op::Op;

std::unordered_map<std::string,
                   std::vector<std::pair<std::string, std::string>>>
    MergeIndependentPass::merge_node_pairs_by_graph;

// std::set<std::pair<std::string, std::string>>
// *MergeIndependentPass::merge_node_pairs() {
//     if (merge_node_pairs_ == nullptr) {
//         merge_node_pairs_ = new std::set<std::pair<std::string,
//         std::string>>;
//     }

//     return merge_node_pairs_;
// }

std::string MergeIndependentPass::name() const {
  return PassName::MergeIndependentPass;
}

void MergeIndependentPass::do_merge(std::shared_ptr<Op>& node1,
                                    std::shared_ptr<Op>& node2, Graph* graph) {
  std::string node1_name = node1->get_name();
  std::string node2_name = node2->get_name();

  std::shared_ptr<ClusterOp> new_cluster_node;
  std::string new_cluster_node_name =
      node1_name + "_" + this->name() + "_" + node2_name;

  if (node1->as<ClusterOp>()->is_cluster_reduce() ||
      node2->as<ClusterOp>()->is_cluster_reduce()) {
    new_cluster_node = std::static_pointer_cast<ClusterOp>(
        ClusterUtil::merge_independent<ClusterReduce>(
            new_cluster_node_name, std::static_pointer_cast<ClusterOp>(node1),
            std::static_pointer_cast<ClusterOp>(node2)));
  } else {
    new_cluster_node = std::static_pointer_cast<ClusterOp>(
        ClusterUtil::merge_independent<ClusterElewise>(
            new_cluster_node_name, std::static_pointer_cast<ClusterOp>(node1),
            std::static_pointer_cast<ClusterOp>(node2)));
  }

  new_cluster_node->set_horizontal_fusion_count(
      node1->as<ClusterOp>()->get_horizontal_fusion_count() +
      node2->as<ClusterOp>()->get_horizontal_fusion_count());

  ClusterUtil::replace_node(graph, {node1_name, node2_name}, new_cluster_node);

  LOG(INFO) << "Pass: " << this->name() << ". Merge " << node1_name << " and "
            << node2_name << " into " << new_cluster_node_name;
}

bool MergeIndependentPass::run(Graph* graph,
                               std::shared_ptr<CUDAContext> cuda_context) {
  if (graph->get_graph_name().empty()) {
    LOG(FATAL) << "Graph name empty.";
  }

  if (MergeIndependentPass::merge_node_pairs_by_graph.count(
          graph->get_graph_name())) {
    for (auto const& [node1_name, node2_name] : MergeIndependentPass::
             merge_node_pairs_by_graph[graph->get_graph_name()]) {
      std::shared_ptr<Op> node1 = graph->get_node(node1_name);
      std::shared_ptr<Op> node2 = graph->get_node(node2_name);
      this->do_merge(node1, node2, graph);
    }

    return true;
  }

  MergeIndependentPass::merge_node_pairs_by_graph[graph->get_graph_name()] =
      std::vector<std::pair<std::string, std::string>>();

  while (true) {
    graph->build_transitive_closure();
    std::vector<std::string> cluster_node_list =
        graph->get_node_list_by_type(OpType::cluster);
    bool found_node_pair = false;

    for (int idx1 = 0; idx1 < (int)cluster_node_list.size(); ++idx1) {
      for (int idx2 = idx1 + 1; idx2 < (int)cluster_node_list.size(); ++idx2) {
        std::string node1_name = cluster_node_list[idx1];
        std::string node2_name = cluster_node_list[idx2];

        std::shared_ptr<Op> node1 = graph->get_node(node1_name);
        std::shared_ptr<Op> node2 = graph->get_node(node2_name);

        if (graph->topology_before(node1_name, node2_name) ||
            graph->topology_before(node2_name, node1_name))
          continue;

        if (node1->as<ClusterOp>()->get_loop_shape().element_count() !=
            node2->as<ClusterOp>()->get_loop_shape().element_count())
          continue;

        if (!node1->as<ClusterOp>()->is_cluster_elewise() &&
            !node1->as<ClusterOp>()->is_cluster_reduce())
          continue;

        if (!node2->as<ClusterOp>()->is_cluster_elewise() &&
            !node2->as<ClusterOp>()->is_cluster_reduce())
          continue;

        if (node1->is_cluster_reduce() && node2->is_cluster_reduce() &&
            node1->as<ClusterReduce>()->get_reduction_dimension_size() !=
                node2->as<ClusterReduce>()->get_reduction_dimension_size()) {
          continue;
        }

        if (node1->as<ClusterOp>()->get_horizontal_fusion_count() +
                node2->as<ClusterOp>()->get_horizontal_fusion_count() >
            6) {
          continue;
        }

        if (node2->as<ClusterOp>()->is_cluster_reduce()) {
          std::swap(node1, node2);
          std::swap(node1_name, node2_name);
        }

        MergeIndependentPass::merge_node_pairs_by_graph[graph->get_graph_name()]
            .push_back(std::make_pair(node1_name, node2_name));

        found_node_pair = true;
        this->do_merge(node1, node2, graph);

        break;

        // std::shared_ptr<ClusterOp> new_cluster_node;
        // std::string new_cluster_node_name = node1_name + "_" + this->name() +
        // "_" + node2_name;

        // if (node1->as<ClusterOp>()->is_cluster_reduce() ||
        //     node2->as<ClusterOp>()->is_cluster_reduce()) {
        //     new_cluster_node =
        //     std::static_pointer_cast<ClusterOp>(ClusterUtil::merge_independent<ClusterReduce>(
        //             new_cluster_node_name,
        //             std::static_pointer_cast<ClusterOp>(node1),
        //             std::static_pointer_cast<ClusterOp>(node2)));
        // } else {
        //     new_cluster_node =
        //     std::static_pointer_cast<ClusterOp>(ClusterUtil::merge_independent<ClusterElewise>(
        //             new_cluster_node_name,
        //             std::static_pointer_cast<ClusterOp>(node1),
        //             std::static_pointer_cast<ClusterOp>(node2)));
        // }

        // ClusterUtil::replace_node(graph, {node1_name, node2_name},
        // new_cluster_node);

        // LOG(INFO) << "Pass: " << this->name() << ". Merge " << node1_name <<
        // " and " << node2_name << " into " << new_cluster_node_name;
      }

      if (found_node_pair) {
        break;
      }
    }

    if (!found_node_pair) {
      break;
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine
