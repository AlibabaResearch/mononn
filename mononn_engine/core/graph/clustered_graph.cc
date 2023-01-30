#include "mononn_engine/core/graph/clustered_graph.h"

#include <queue>

#include "mononn_engine/core/op/all_cluster_operators.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace graph {
using Op = mononn_engine::core::op::Op;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using Edge = mononn_engine::core::edge::Edge<ClusterOp>;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;

using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using ClusterGemm = mononn_engine::core::op::ClusterGemm;
using ClusterConv = mononn_engine::core::op::ClusterConv;
using ClusterGemmEpilogue = mononn_engine::core::op::ClusterGemmEpilogue;
using ClusterConvEpilogue = mononn_engine::core::op::ClusterConvEpilogue;

void ClusteredGraph::from_graph(std::shared_ptr<Graph> graph) {
  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    if (node->get_cluster_type() == ClusterType::None) {
      LOG(FATAL) << "Node " << node_name << " does not belone to any cluster";
    }
  }

  std::unordered_set<std::string> cluster_name_visit;

  // Add node for clustered graph;
  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    std::string cluster_name = node->get_cluster_name();
    ClusterType cluster_type = node->get_cluster_type();
    int cluster_id = node->get_cluster_id();

    if (cluster_name_visit.find(cluster_name) != cluster_name_visit.end()) {
      continue;
    }

    cluster_name_visit.insert(cluster_name);

    std::unordered_set<std::string> visit;
    std::unordered_set<std::string> cluster_input_nodes, cluster_output_nodes;

    std::function<void(std::shared_ptr<Op>)> find_input_output_nodes =
        [&](std::shared_ptr<Op> dfs_node) -> void {
      if (visit.find(dfs_node->get_name()) != visit.end()) return;
      if (dfs_node->get_cluster_name() != cluster_name) return;

      visit.insert(dfs_node->get_name());

      if (dfs_node->get_operands().size() == 0) {
        cluster_input_nodes.insert(dfs_node->get_name());
      } else {
        for (auto const& operand : dfs_node->get_operands()) {
          if (operand->get_cluster_name() != cluster_name) {
            cluster_input_nodes.insert(dfs_node->get_name());
            break;
          }
        }
      }

      if (graph->get_node_output_edges(dfs_node->get_name()).size() == 0) {
        cluster_output_nodes.insert(dfs_node->get_name());
      } else {
        for (auto const& edge :
             graph->get_node_output_edges(dfs_node->get_name())) {
          if (edge->get_dst()->get_cluster_name() != cluster_name) {
            cluster_output_nodes.insert(dfs_node->get_name());
            break;
          }
        }
      }

      for (auto const& operand : dfs_node->get_operands())
        find_input_output_nodes(operand);
      for (auto const& edge :
           graph->get_node_output_edges(dfs_node->get_name()))
        find_input_output_nodes(edge->get_dst());
    };

    find_input_output_nodes(node);

    std::shared_ptr<Graph> cluster_subgraph =
        graph->get_subgraph(visit, cluster_input_nodes, cluster_output_nodes);

    std::unordered_set<std::string> operands_set;
    std::vector<TensorSpec> output_specs;

    for (auto const& node_name : cluster_input_nodes) {
      std::shared_ptr<Op> node = cluster_subgraph->get_node(node_name);
      for (auto const& operand : node->get_operands()) {
        if (operand->get_cluster_name() != cluster_name)
          operands_set.insert(operand->get_name());
      }
    }

    std::vector<std::shared_ptr<Op>> operands_list;

    for (auto const& operand : operands_set) {
      operands_list.push_back(graph->get_node(operand));
    }

    for (auto const& node_name : cluster_output_nodes) {
      std::shared_ptr<Op> node = cluster_subgraph->get_node(node_name);
      output_specs.push_back(node->get_output_spec(0));
    }

    std::shared_ptr<ClusterOp> cluster_op;

    if (cluster_type == ClusterType::Elewise) {
      cluster_op =
          std::static_pointer_cast<ClusterOp>(std::make_shared<ClusterElewise>(
              cluster_name, operands_list, output_specs));
    } else if (cluster_type == ClusterType::Reduce) {
      cluster_op =
          std::static_pointer_cast<ClusterOp>(std::make_shared<ClusterReduce>(
              cluster_name, operands_list, output_specs));
    } else if (cluster_type == ClusterType::Gemm) {
      cluster_op =
          std::static_pointer_cast<ClusterOp>(std::make_shared<ClusterGemm>(
              cluster_name, operands_list, output_specs));
    } else if (cluster_type == ClusterType::Conv) {
      cluster_op =
          std::static_pointer_cast<ClusterOp>(std::make_shared<ClusterConv>(
              cluster_name, operands_list, output_specs));
    } else if (cluster_type == ClusterType::GemmEpilogue) {
      cluster_op = std::static_pointer_cast<ClusterOp>(
          std::make_shared<ClusterGemmEpilogue>(cluster_name, operands_list,
                                                output_specs));
    } else if (cluster_type == ClusterType::ConvEpilogue) {
      cluster_op = std::static_pointer_cast<ClusterOp>(
          std::make_shared<ClusterConvEpilogue>(cluster_name, operands_list,
                                                output_specs));
    } else {
      LOG(FATAL) << "Unsupported cluster type " << cluster_type.to_string();
    }

    cluster_op->set_cluster_type(cluster_type);
    cluster_op->set_cluster_id(cluster_id);
    cluster_op->set_graph(cluster_subgraph);

    this->add_node(cluster_op);
  }

  std::vector<std::pair<std::string, std::string>> edge_list;

  // Add edge for clustered graph
  for (auto const& node_name : graph->get_node_list()) {
    std::shared_ptr<Op> node = graph->get_node(node_name);
    std::string cluster_name = node->get_cluster_name();

    for (auto const& edge : graph->get_node_output_edges(node_name)) {
      if (cluster_name != edge->get_dst()->get_cluster_name()) {
        edge_list.push_back(
            std::make_pair(cluster_name, edge->get_dst()->get_cluster_name()));
      }
    }
  }

  // remove duplicate
  std::sort(edge_list.begin(), edge_list.end());
  edge_list.erase(std::unique(edge_list.begin(), edge_list.end()),
                  edge_list.end());

  for (auto const& edge : edge_list) {
    this->add_edge(std::make_shared<Edge>(this->get_node(edge.first),
                                          this->get_node(edge.second)));
  }

  // mark input/output cluster/op of graph
  std::unordered_map<std::string, int> cluster_op_in_degree;
  std::unordered_map<std::string, int> cluster_op_out_degree;

  for (auto const& node_name : this->get_node_list()) {
    cluster_op_in_degree[node_name] = 0;
    cluster_op_out_degree[node_name] = 0;
  }

  for (auto const& node_name : this->get_node_list()) {
    for (auto const& edge : this->get_node_output_edges(node_name)) {
      std::string dst = edge->get_dst()->get_name();
      std::string src = edge->get_src()->get_name();

      cluster_op_in_degree[dst] = cluster_op_in_degree[dst] + 1;
      cluster_op_out_degree[src] = cluster_op_out_degree[src] + 1;
    }
  }

  this->input_nodes.clear();
  this->output_nodes.clear();

  for (auto const& [node_name, in_deg] : cluster_op_in_degree) {
    if (in_deg == 0) this->input_nodes.push_back(node_name);
  }

  for (auto const& [node_name, out_deg] : cluster_op_out_degree) {
    if (out_deg == 0) this->output_nodes.push_back(node_name);
  }
}

void ClusteredGraph::add_node(std::shared_ptr<ClusterOp> node) {
  std::string node_name = node->get_name();

  if (this->nodes.find(node_name) != this->nodes.end()) {
    LOG(FATAL) << "Duplicate node: " << node_name;
  }

  this->nodes[node_name] = node;
  this->edges[node_name] = std::vector<std::shared_ptr<Edge>>();
}

void ClusteredGraph::add_edge(std::shared_ptr<Edge> edge) {
  std::string node_name = edge->get_src()->get_name();

  if (this->edges.find(node_name) == this->edges.end()) {
    this->edges[node_name] = std::vector<std::shared_ptr<Edge>>();
  }

  this->edges[node_name].push_back(edge);
}

std::shared_ptr<ClusterOp> ClusteredGraph::get_node(std::string node) {
  return this->nodes[node];
}

std::vector<std::string> ClusteredGraph::get_node_list() const {
  std::vector<std::string> node_list;

  for (auto const& [node_name, node] : this->nodes) {
    node_list.push_back(node_name);
  }

  return node_list;
}

std::vector<std::string> ClusteredGraph::get_input_nodes() const {
  return this->input_nodes;
}

std::vector<std::string> ClusteredGraph::get_output_nodes() const {
  return this->output_nodes;
}

std::vector<std::shared_ptr<Edge>> ClusteredGraph::get_node_output_edges(
    std::string node) {
  return this->edges[node];
}

void ClusteredGraph::wave_front_order(
    std::function<void(std::shared_ptr<ClusterOp>)> func) {
  std::unordered_map<std::string, int> steps;

  for (auto const& node_name : this->get_node_list()) {
    steps[node_name] = -1;
  }

  std::queue<std::pair<std::string, int>> q;

  for (auto const& node_name : this->get_input_nodes()) {
    q.push(std::make_pair(node_name, 0));
  }

  while (!q.empty()) {
    std::pair<std::string, int> n = q.front();
    q.pop();

    std::string& node_name = n.first;
    int step = n.second;

    if (step > steps[node_name]) {
      steps[node_name] = step;

      for (auto const& edge : this->get_node_output_edges(node_name)) {
        q.push(std::make_pair(edge->get_dst()->get_name(), step + 1));
      }
    }
  }

  std::vector<std::pair<std::string, int>> traverse_order;

  for (auto const& [node_name, step] : steps) {
    traverse_order.push_back(std::make_pair(node_name, step));
  }

  std::sort(traverse_order.begin(), traverse_order.end(),
            [](std::pair<std::string, int> const& a,
               std::pair<std::string, int> const& b) -> bool {
              return a.second < b.second;
            });

  for (auto const& node : traverse_order) {
    func(this->get_node(node.first));
  }
}

std::string ClusteredGraph::summary() {
  std::stringstream ss;
  ss << "Summary of clustered graph:\n";
  for (auto const& clustered_node_name : this->get_node_list()) {
    ss << "     Cluster:" << clustered_node_name;
    for (auto const& node_name :
         this->get_node(clustered_node_name)->get_graph()->get_node_list()) {
      ss << " " << node_name;
    }

    ss << "\n";
  }

  return ss.str();
}
}  // namespace graph
}  // namespace core
}  // namespace mononn_engine