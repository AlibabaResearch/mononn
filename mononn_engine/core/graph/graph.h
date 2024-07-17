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

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mononn_engine/core/context/index_trace_stamp.h"
#include "mononn_engine/core/edge/control_edge.h"
#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace core {
namespace graph {
class Graph {
 public:
  using Op = mononn_engine::core::op::Op;
  using Edge = mononn_engine::core::edge::Edge<Op>;
  using ClusterType = mononn_engine::core::op_annotation::ClusterType;
  using ControlEdge = mononn_engine::core::edge::ControlEdge;
  using OpType = mononn_engine::core::op::OpType;
  using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

  Graph(){};

  void set_graph_name(const std::string& _graph_name);
  const std::string& get_graph_name() const;

  void add_node(std::shared_ptr<Op>& node);
  void add_node(std::shared_ptr<Op>&& node);
  void remove_node(std::shared_ptr<Op> node);
  void remove_node(const std::string& node_name);
  void add_edge(std::shared_ptr<Edge> edge);
  void add_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst);
  void add_edge(const std::string& node_src, const std::string& node_dst);
  void remove_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst);
  void remove_edge(const std::string& src, const std::string& dst);
  void remove_edge(std::shared_ptr<Edge> edge);
  void remove_edge_if(std::function<bool(std::shared_ptr<Edge> const)> pred);

  void add_control_edge(const std::string& node_src,
                        const std::string& node_dst);
  void add_control_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst);
  void add_control_edge(std::shared_ptr<ControlEdge> control_edge);
  void remove_control_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst);
  void replace_node(const std::vector<std::string>& node_list,
                    std::shared_ptr<Op> new_node);

  bool has_node(const std::string& node_name) const;
  bool has_edge(const std::string& node_src, const std::string& node_dst) const;
  bool has_control_edge(const std::string& node_src,
                        const std::string& node_dst) const;

  std::shared_ptr<Op> get_node(const std::string& node);
  std::shared_ptr<const Op> get_node(const std::string& node) const;
  std::vector<std::string> get_node(
      std::function<bool(const Op* op)> pred) const;

  std::shared_ptr<Edge> get_edge(const std::string& node_src,
                                 const std::string& node_dst);

  std::vector<std::string> get_node_list() const;
  std::vector<std::string> get_node_list_by_type(const OpType& op_type) const;

  int get_node_num(OpType op_type) const;

  std::vector<std::shared_ptr<Edge>> get_node_output_edges(
      const std::string& node);
  std::vector<std::shared_ptr<Edge>> const get_node_output_edges(
      const std::string& node) const;
  std::vector<std::shared_ptr<ControlEdge>> get_node_output_control_edges(
      const std::string& node);
  std::vector<std::shared_ptr<ControlEdge>> const get_node_output_control_edges(
      const std::string& node) const;

  std::vector<std::shared_ptr<Edge>> get_node_input_edges(
      const std::string& node) const;
  std::vector<std::shared_ptr<ControlEdge>> get_node_input_control_edges(
      const std::string& node) const;

  void mark_as_input_node(const std::string& node);
  const std::vector<std::string>& get_input_nodes() const;
  std::string get_input_node(int idx) const;
  int get_input_node_count() const;
  void sort_input_nodes();
  void align_input_nodes();

  bool is_input_node(const std::string& node_name) const;

  void mark_as_extended_input_node(const std::string& node);
  const std::vector<std::string>& get_extended_input_nodes() const;

  void mark_as_output_node(const std::string& node);
  const std::vector<std::string>& get_output_nodes() const;
  std::string get_output_node(int idx) const;
  int get_output_node_count() const;
  int find_output_node_idx(const std::string& node_name) const;
  bool is_output_node(const std::string& node_name) const;

  void update_output_node(int idx, const std::string& new_node);

  void verify();

  void set_node_attribute(std::shared_ptr<Op> node, const std::string& key,
                          const std::string& value);
  void set_node_attribute(const std::string& node_name, const std::string& key,
                          const std::string& value);
  std::string get_node_attribute(std::shared_ptr<Op> node,
                                 const std::string& key) const;
  std::string get_node_attribute(const std::string& node_name,
                                 const std::string& key) const;
  bool node_has_attribute(std::shared_ptr<Op> node,
                          const std::string& key) const;
  bool node_has_attribute(const std::string& node_name,
                          const std::string& key) const;

  // Graph traversal algorithms
  std::unordered_set<std::string> bfs(
      const std::unordered_set<std::string>& start_nodes);
  std::unordered_set<std::string> bfs(
      const std::unordered_set<std::string>& start_nodes,
      std::function<void(std::shared_ptr<Op>)> func);

  std::unordered_set<std::string> post_order(const std::string& start_node);
  std::unordered_set<std::string> post_order(
      const std::string& start_node,
      std::function<void(std::shared_ptr<Op>)> func);

  std::unordered_set<std::string> post_order_visit_all_nodes(
      std::function<void(std::shared_ptr<Op>)> pre_process,
      std::function<void(std::shared_ptr<Op>)> func);

  std::unordered_set<std::string> reverse_post_order_visit_all_nodes(
      std::function<void(std::shared_ptr<Op>)> pre_process,
      std::function<void(std::shared_ptr<Op>)> func);

  std::vector<std::string> traverse_in_topology_order() const;
  void wave_front_order(
      std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func);
  void wave_front_order(
      std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
          func) const;
  void topology_order(
      std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func);
  void topology_order(
      std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
          func) const;
  void reverse_topology_order(
      std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func);
  void reverse_topology_order(
      std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
          func) const;

  // End graph traversal algorithms
  std::string summary() const;
  void neo4j_summary(std::string file, std::string tag) const;

  std::shared_ptr<Graph> get_subgraph(
      const std::unordered_set<std::string>& subgraph_node_list,
      const std::unordered_set<std::string>& subgraph_input_nodes,
      const std::unordered_set<std::string>& subgraph_output_nodes);

  void clustering();
  bool is_clustered() const;

  void build_transitive_closure(bool including_control_dependency = true);
  bool topology_before(const std::string& node1,
                       const std::string& node2) const;
  int distance(const std::string& node_src, const std::string& node_dst) const;
  bool reachable_under_constrain(const std::string& node_src,
                                 const std::string& node_dst,
                                 const OpType& op_type_constrain) const;
  std::vector<std::string> search_downstream_node_under_constrain(
      const std::string& node_src,
      std::function<bool(const Op*)> path_constrain,
      std::function<bool(const Op*, std::vector<std::string> const&)>
          dst_constrain,
      bool return_nodes_in_path = false) const;

  std::vector<std::string> search_downstream_node_under_constrain(
      const std::string& node_src,
      std::function<bool(const Op*, std::vector<std::string> const&)>
          dst_constrain,
      bool return_nodes_in_path = false) const;

  std::vector<std::string> search_nodes_between_two_nodes(
      const std::string& node_src, const std::string& node_dst) const;

  void trace_index(const std::string& index, const std::string& node_name,
                   std::string inverse_reduce_dimension = "");
  //        void trace_ilp_index(int ilp_id, const std::string &index, const
  //        std::string &node_name, std::string inverse_reduce_dimension = "")
  //        override;

  const std::vector<SymbolicIndexStamp>& get_symbolic_index(
      const std::string& node_name) const;
  bool is_node_traced(const std::string& node_name) const;

  void reset_symbolic_index();

  std::vector<std::string> get_symbolic_index_before_trace(
      const std::string& node_name) const;
  std::vector<std::string> get_symbolic_index_after_trace(
      const std::string& node_name) const;

  bool is_simple_padding_scenario() const;
  bool is_dead_node(const std::string& node_name) const;

  bool remain_acyclic_after_node_merge(
      const std::vector<std::string>& node_list) const;

  void set_instruction_parallel_factor(int _ilp_factor);
  int get_instruction_parallel_factor() const;

  bool is_acyclic(std::vector<std::string>& nodes_in_cycle,
                  bool include_control_edge = false) const;
  void verify_is_acyclic(bool include_control_edge = false) const;

  // std::unique_ptr<Graph> deep_copy() const;

 private:
  std::string graph_name;
  std::unordered_map<std::string, std::shared_ptr<Op>> nodes;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Edge>>> edges;
  std::unordered_map<std::string, std::vector<std::shared_ptr<ControlEdge>>>
      control_edges;

  std::unordered_map<std::string, std::vector<SymbolicIndexStamp>>
      symbolic_index;

  // input nodes that read from global memory
  std::vector<std::string> input_nodes;
  // other input nodes (e.g. iota, constant scalar)
  std::vector<std::string> extended_input_nodes;
  std::vector<std::string> output_nodes;
  int reduce_cluster_id;
  int elewise_cluster_id;
  int gemm_epilogue_cluster_id;
  int conv_epilogue_cluster_id;
  int gemm_cluster_id;
  int conv_cluster_id;
  int ilp_factor = 1;

  template <typename T1, typename T2>
  struct pair_hash {
    inline std::size_t operator()(std::pair<T1, T2> const& v) const noexcept {
      std::size_t hash1 = std::hash<T1>{}(v.first);
      std::size_t hash2 = std::hash<T2>{}(v.second);

      return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
  };

  std::unordered_map<std::pair<std::string, std::string>, int,
                     pair_hash<std::string, std::string>>
      transitive_closure;

  void post_order_impl(std::string start_node,
                       std::function<void(std::shared_ptr<Op>)> func,
                       std::unordered_set<std::string>& visit);
  void post_order_impl(std::string start_node,
                       std::unordered_set<std::string>& visit,
                       std::function<void(std::shared_ptr<Op>)> pre_process,
                       std::function<void(std::shared_ptr<Op>)> func);

  void reverse_post_order_impl(
      std::string start_node, std::unordered_set<std::string>& visit,
      std::function<void(std::shared_ptr<Op>)> pre_process,
      std::function<void(std::shared_ptr<Op>)> func);

  void cluster_reduce(std::shared_ptr<Op> node);
  void cluster_elewise(std::shared_ptr<Op> node);
  void cluster_gemm_epilogue(std::shared_ptr<Op> node);
  void cluster_conv_epilogue(std::shared_ptr<Op> node);
  void cluster_gemm(std::shared_ptr<Op> node);
  void cluster_conv(std::shared_ptr<Op> node);

  bool remain_acyclic_after_add_node_to_cluster(std::shared_ptr<Op> node_to_add,
                                                ClusterType cluster_type,
                                                int cluster_id);
};
}  // namespace graph
}  // namespace core
}  // namespace mononn_engine