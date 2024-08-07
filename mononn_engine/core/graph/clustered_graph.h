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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/op.h"

namespace mononn_engine {
namespace core {
namespace graph {
class ClusteredGraph {
 public:
  using Op = mononn_engine::core::op::Op;
  using ClusterOp = mononn_engine::core::op::ClusterOp;
  using Edge = mononn_engine::core::edge::Edge<ClusterOp>;
  using Graph = mononn_engine::core::graph::Graph;

  ClusteredGraph(){};

  void from_graph(std::shared_ptr<Graph> graph);

  std::shared_ptr<ClusterOp> get_node(std::string node);
  std::vector<std::string> get_node_list() const;
  std::vector<std::shared_ptr<Edge>> get_node_output_edges(std::string node);

  std::vector<std::string> get_input_nodes() const;
  std::vector<std::string> get_output_nodes() const;

  void wave_front_order(std::function<void(std::shared_ptr<ClusterOp>)> func);

  std::string summary();

 private:
  std::unordered_map<std::string, std::shared_ptr<ClusterOp>> nodes;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Edge>>> edges;

  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;

  void add_node(std::shared_ptr<ClusterOp> node);
  void add_edge(std::shared_ptr<Edge> edge);
};
}  // namespace graph
}  // namespace core
}  // namespace mononn_engine