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

#include "mononn_engine/core/graph/graph.h"

#include <functional>
#include <map>
#include <numeric>
#include <queue>

#include "mononn_engine/core/context/index_tracer.h"
#include "mononn_engine/core/op/concatenate.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/custom_call.h"
#include "mononn_engine/core/op/dynamic_slice.h"
#include "mononn_engine/core/op/dynamic_update_slice.h"
#include "mononn_engine/core/op/gather.h"
#include "mononn_engine/core/op/get_tuple_element.h"
#include "mononn_engine/core/op/pad.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/op/reduce.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/macros.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace graph {
using Op = mononn_engine::core::op::Op;
using Edge = mononn_engine::core::edge::Edge<Op>;
using OpType = mononn_engine::core::op::OpType;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using CustomCall = mononn_engine::core::op::CustomCall;
using ControlEdge = mononn_engine::core::edge::ControlEdge;
using DynamicSlice = mononn_engine::core::op::DynamicSlice;
using DynamicUpdateSlice = mononn_engine::core::op::DynamicUpdateSlice;
using IndexTracer = mononn_engine::core::context::IndexTracer;
using Gather = mononn_engine::core::op::Gather;
using Reduce = mononn_engine::core::op::Reduce;
using Concatenate = mononn_engine::core::op::Concatenate;
using Constant = mononn_engine::core::op::Constant;
using Parameter = mononn_engine::core::op::Parameter;
using Pad = mononn_engine::core::op::Pad;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;
using GetTupleElement = mononn_engine::core::op::GetTupleElement;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;

void Graph::set_graph_name(const std::string& _graph_name) {
  this->graph_name = _graph_name;
}

const std::string& Graph::get_graph_name() const { return this->graph_name; }

void Graph::add_node(std::shared_ptr<Op>& node) {
  std::string node_name = node->get_name();

  if (this->nodes.find(node_name) != this->nodes.end()) {
    LOG(FATAL) << "Duplicate node: " << node_name;
  }

  this->nodes[node_name] = node;

  if (this->edges.find(node_name) == this->edges.end()) {
    this->edges[node_name] = std::vector<std::shared_ptr<Edge>>();
  }
}

void Graph::add_node(std::shared_ptr<Op>&& node) {
  std::string node_name = node->get_name();

  if (this->nodes.find(node_name) != this->nodes.end()) {
    LOG(FATAL) << "Duplicate node: " << node_name;
  }

  this->nodes[node_name] = node;

  if (this->edges.find(node_name) == this->edges.end()) {
    this->edges[node_name] = std::vector<std::shared_ptr<Edge>>();
  }
}

void Graph::remove_node(std::shared_ptr<Op> node) {
  std::string node_name = node->get_name();
  if (this->get_node_input_edges(node_name).size() != 0 ||
      this->get_node_output_edges(node_name).size() != 0 ||
      this->get_node_input_control_edges(node_name).size() != 0 ||
      this->get_node_output_control_edges(node_name).size() != 0) {
    LOG(FATAL) << "Cannot remove node " << node_name << " from graph.\n"
               << "node input edge count: "
               << this->get_node_input_edges(node_name).size() << "\n"
               << "node output edge count: "
               << this->get_node_output_edges(node_name).size() << "\n"
               << "node input control edge count: "
               << this->get_node_input_control_edges(node_name).size() << "\n"
               << "node output control edge count: "
               << this->get_node_output_control_edges(node_name).size();
  }

  this->nodes.erase(node_name);
  if (this->symbolic_index.count(node_name) > 0)
    this->symbolic_index.erase(node_name);
  std::vector<std::string>::iterator iter =
      std::find(this->input_nodes.begin(), this->input_nodes.end(), node_name);
  if (iter != this->input_nodes.end()) this->input_nodes.erase(iter);

  iter = std::find(this->extended_input_nodes.begin(),
                   this->extended_input_nodes.end(), node_name);
  if (iter != this->extended_input_nodes.end())
    this->extended_input_nodes.erase(iter);

  iter = std::find(this->output_nodes.begin(), this->output_nodes.end(),
                   node_name);
  if (iter != this->output_nodes.end()) this->output_nodes.erase(iter);
}

void Graph::remove_node(const std::string& node_name) {
  this->remove_node(this->get_node(node_name));
}

void Graph::add_edge(std::shared_ptr<Edge> edge) {
  EXPECT_TRUE(edge->get_src_name() != edge->get_dst_name(),
              "Self cycle is not allowed");
  std::string node_name = edge->get_src()->get_name();

  if (this->edges.find(node_name) == this->edges.end()) {
    this->edges[node_name] = std::vector<std::shared_ptr<Edge>>();
  }

  this->edges[node_name].push_back(edge);
}

void Graph::add_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst) {
  EXPECT_TRUE(src->get_name() != dst->get_name(), "Self cycle is not allowed");

  std::shared_ptr<Edge> edge = std::make_shared<Edge>(src, dst);

  this->add_edge(edge);
}

void Graph::add_edge(const std::string& node_src, const std::string& node_dst) {
  EXPECT_TRUE(node_src != node_dst, "Self cycle is not allowed");

  this->add_edge(this->get_node(node_src), this->get_node(node_dst));
}

void Graph::remove_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst) {
  if (this->edges.count(src->get_name()) == 0) {
    LOG(FATAL) << "Edge not found";
  }

  std::vector<std::shared_ptr<Edge>>::iterator iter = std::find_if(
      this->edges[src->get_name()].begin(), this->edges[src->get_name()].end(),
      [&](std::shared_ptr<Edge>& e) -> bool {
        return e->get_src()->get_name() == src->get_name() &&
               e->get_dst()->get_name() == dst->get_name();
      });

  if (iter == this->edges[src->get_name()].end()) {
    LOG(FATAL) << "Edge not found";
  }

  this->edges[src->get_name()].erase(iter);
}

void Graph::remove_edge(std::shared_ptr<Edge> edge) {
  this->remove_edge(edge->get_src(), edge->get_dst());
}

void Graph::remove_edge(const std::string& src, const std::string& dst) {
  this->remove_edge(this->get_node(src), this->get_node(dst));
}

void Graph::remove_edge_if(
    std::function<bool(std::shared_ptr<Edge> const)> pred) {
  std::vector<std::pair<std::string, std::string>> edge_to_be_removed;

  for (auto const& [node_name, edge_list] : this->edges) {
    for (auto const& edge : edge_list) {
      if (pred(edge)) {
        edge_to_be_removed.push_back(
            std::make_pair(edge->get_src_name(), edge->get_dst_name()));
      }
    }
  }

  for (auto const& [node_src, node_dst] : edge_to_be_removed) {
    this->remove_edge(node_src, node_dst);
  }
}

void Graph::add_control_edge(std::shared_ptr<ControlEdge> control_edge) {
  EXPECT_TRUE(control_edge->get_src_name() != control_edge->get_dst_name(),
              "Self cycle is not allowed");

  if (this->nodes.find(control_edge->get_src()->get_name()) ==
      this->nodes.end()) {
    LOG(FATAL) << "Node: " << control_edge->get_src()->get_name()
               << " not in graph";
  }

  if (this->nodes.find(control_edge->get_dst()->get_name()) ==
      this->nodes.end()) {
    LOG(FATAL) << "Node: " << control_edge->get_dst()->get_name()
               << " not in graph";
  }

  if (this->control_edges.find(control_edge->get_src()->get_name()) ==
      this->control_edges.end()) {
    this->control_edges[control_edge->get_src()->get_name()] =
        std::vector<std::shared_ptr<ControlEdge>>();
  }

  this->control_edges[control_edge->get_src()->get_name()].push_back(
      control_edge);
}

void Graph::add_control_edge(const std::string& node_src,
                             const std::string& node_dst) {
  EXPECT_TRUE(node_src != node_dst, "Self cycle is not allowed");
  this->add_control_edge(this->get_node(node_src), this->get_node(node_dst));
}

void Graph::add_control_edge(std::shared_ptr<Op> src, std::shared_ptr<Op> dst) {
  EXPECT_TRUE(src->get_name() != dst->get_name(), "Self cycle is not allowed");
  std::shared_ptr<ControlEdge> control_edge =
      std::make_shared<ControlEdge>(src, dst);
  this->add_control_edge(control_edge);
}

void Graph::remove_control_edge(std::shared_ptr<Op> src,
                                std::shared_ptr<Op> dst) {
  if (this->control_edges.count(src->get_name()) == 0) {
    LOG(FATAL) << "Edge not found";
  }

  std::vector<std::shared_ptr<ControlEdge>>::iterator iter =
      std::find_if(this->control_edges[src->get_name()].begin(),
                   this->control_edges[src->get_name()].end(),
                   [&](std::shared_ptr<ControlEdge>& e) -> bool {
                     return e->get_src()->get_name() == src->get_name() &&
                            e->get_dst()->get_name() == dst->get_name();
                   });

  if (iter == this->control_edges[src->get_name()].end()) {
    LOG(FATAL) << "Edge not found";
  }

  this->control_edges[src->get_name()].erase(iter);
}

void Graph::replace_node(const std::vector<std::string>& node_list,
                         std::shared_ptr<Op> new_node) {
  LOG(WARNING) << "Deprecated, use ClusterUtil::replace_node instead";
  this->add_node(new_node);

  if (std::any_of(node_list.begin(), node_list.end(),
                  [&](std::string const& node_name) -> bool {
                    return this->is_output_node(node_name);
                  })) {
    LOG(FATAL) << "Mark output node: " << new_node->get_name();
    this->mark_as_output_node(new_node->get_name());
  }

  for (auto const& node_name : node_list) {
    std::vector<std::string>::iterator iter = std::find(
        this->input_nodes.begin(), this->input_nodes.end(), node_name);
    if (iter != this->input_nodes.end()) {
      LOG(FATAL) << "Cannot replace input nodes";
    }

    iter = std::find(this->extended_input_nodes.begin(),
                     this->extended_input_nodes.end(), node_name);
    if (iter != this->extended_input_nodes.end()) {
      LOG(FATAL) << "Cannot replace extended input nodes";
    }

    std::shared_ptr<Op> node = this->get_node(node_name);
    EXPECT_TRUE(node->get_type() == OpType::cluster ||
                    node->get_type() == OpType::get_tuple_element,
                "Node " + node_name +
                    " is neither cluster nor get tuple element node.");
  }

  std::function<bool(std::string, std::string)> edge_in_node_list =
      [&](std::string node_name_src, std::string node_name_dst) -> bool {
    return std::find(node_list.begin(), node_list.end(), node_name_src) !=
               node_list.end() &&
           std::find(node_list.begin(), node_list.end(), node_name_dst) !=
               node_list.end();
  };

  std::string new_node_name = new_node->get_name();
  int get_tuple_element_id = 0;

  std::function<std::string()> next_get_tuple_element_name =
      [&]() -> std::string {
    return "get_tuple_element_" + new_node_name + "_" +
           std::to_string(get_tuple_element_id++);
  };

  std::vector<std::string> get_tuple_element_node_to_be_removed;
  for (auto const& node_name : node_list) {
    if (this->get_node(node_name)->get_type() == OpType::get_tuple_element)
      continue;

    std::vector<std::shared_ptr<Edge>> node_input_edges =
        this->get_node_input_edges(node_name);
    std::vector<std::shared_ptr<Edge>> node_output_edges =
        this->get_node_output_edges(node_name);
    std::vector<std::shared_ptr<ControlEdge>> node_input_control_edges =
        this->get_node_input_control_edges(node_name);
    std::vector<std::shared_ptr<ControlEdge>> node_output_control_edges =
        this->get_node_output_control_edges(node_name);

    for (auto const& edge : node_input_edges) {
      this->remove_edge(edge->get_src(), edge->get_dst());

      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<Edge> new_edge =
          std::make_shared<Edge>(edge->get_src(), new_node);
      new_edge->set_sync(edge->get_sync());
      this->add_edge(new_edge);
    }

    for (auto const& edge : node_input_control_edges) {
      this->remove_control_edge(edge->get_src(), edge->get_dst());
      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<ControlEdge> new_control_edge =
          std::make_shared<ControlEdge>(edge->get_src(), new_node);
      this->add_control_edge(new_control_edge);
    }

    for (auto const& edge : node_output_control_edges) {
      this->remove_control_edge(edge->get_src(), edge->get_dst());
      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<ControlEdge> new_control_edge =
          std::make_shared<ControlEdge>(new_node, edge->get_dst());
      this->add_control_edge(new_control_edge);
    }

    if (this->get_node(node_name)->get_output_specs().size() > 1) {
      for (auto const& edge :
           node_output_edges) {  // for each get tuple element
        if (edge->get_dst()->get_type() != OpType::get_tuple_element) {
          LOG(FATAL) << "Node " << node_name
                     << " have multiple output spec but "
                     << edge->get_dst()->get_name()
                     << " is not get tuple element";
        }

        std::vector<std::shared_ptr<Edge>> old_get_tuple_element_output_edges =
            this->get_node_output_edges(edge->get_dst()->get_name());

        if (std::all_of(
                old_get_tuple_element_output_edges.begin(),
                old_get_tuple_element_output_edges.end(),
                [&](std::shared_ptr<Edge> get_tuple_element_output_edge)
                    -> bool {
                  return std::find(
                             node_list.begin(), node_list.end(),
                             get_tuple_element_output_edge->get_dst_name()) !=
                         node_list.end();
                })) {
          for (auto const& get_tuple_element_output_edge :
               old_get_tuple_element_output_edges) {
            this->remove_edge(get_tuple_element_output_edge);
          }
        } else {
          // node specified by node_name can be
          // 1. participate in independent merge with multiple output
          // 2. first node in dependent merge with multiple output, as least one
          // of them are output node
          // 3. second node in dependemt merge with multiple output
          // In all cases, new node will have multiple outputs
          EXPECT_TRUE(new_node->get_output_specs_count() > 1,
                      "Node " + new_node->get_name() +
                          " should have multiple output specs");

          std::shared_ptr<GetTupleElement> get_tuple_element =
              std::make_shared<GetTupleElement>(
                  next_get_tuple_element_name(),
                  std::vector<std::shared_ptr<Op>>{new_node},
                  std::vector<TensorSpec>{edge->get_dst()->get_output_spec(0)});
          get_tuple_element->set_tuple_index(get_tuple_element_id - 1);
          this->add_node(get_tuple_element);
          this->add_edge(new_node, get_tuple_element);

          for (auto const& old_get_tuple_element_output_edge :
               old_get_tuple_element_output_edges) {
            if (edge_in_node_list(
                    old_get_tuple_element_output_edge->get_src_name(),
                    old_get_tuple_element_output_edge->get_dst_name())) {
              this->remove_edge(old_get_tuple_element_output_edge);
              continue;
            }

            this->add_edge(get_tuple_element,
                           old_get_tuple_element_output_edge->get_dst());
            old_get_tuple_element_output_edge->get_dst()->replace_operand(
                old_get_tuple_element_output_edge->get_src()->get_name(),
                get_tuple_element);
            this->remove_edge(old_get_tuple_element_output_edge);
          }
        }

        this->remove_edge(edge);
        if (std::find(node_list.begin(), node_list.end(),
                      edge->get_dst_name()) == node_list.end()) {
          this->remove_node(edge->get_dst());
        }
      }
    } else {
      // all output edges are in the node list to be merged
      if (std::all_of(node_output_edges.begin(), node_output_edges.end(),
                      [&](std::shared_ptr<Edge> edge) -> bool {
                        return std::find(node_list.begin(), node_list.end(),
                                         edge->get_dst_name()) !=
                               node_list.end();
                      })) {
        for (auto const& edge : node_output_edges) {
          this->remove_edge(edge);
        }
      } else {
        // No get tuple element node is needed if two cluster is in below
        // format:
        // A
        // |
        // B
        // All output of A goes in B and B only have one output tensor (i.e. not
        // tuple)
        if (new_node->get_output_specs_count() > 1) {
          std::shared_ptr<GetTupleElement> get_tuple_element =
              std::make_shared<GetTupleElement>(
                  next_get_tuple_element_name(),
                  std::vector<std::shared_ptr<Op>>{new_node},
                  std::vector<TensorSpec>{
                      this->get_node(node_name)->get_output_spec(0)});
          get_tuple_element->set_tuple_index(get_tuple_element_id - 1);
          this->add_node(get_tuple_element);
          this->add_edge(new_node, get_tuple_element);

          for (auto const& edge : node_output_edges) {
            if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name())) {
              this->remove_edge(edge);
              continue;
            }

            this->add_edge(get_tuple_element, edge->get_dst());
            edge->get_dst()->replace_operand(node_name, get_tuple_element);
            this->remove_edge(edge);
          }
        } else {
          for (auto const& edge : node_output_edges) {
            EXPECT_FALSE(
                edge_in_node_list(edge->get_src_name(), edge->get_dst_name()),
                "Edge " + edge->to_string() + " should not in node list: " +
                    mononn_engine::helpers::join(" ", node_list));

            this->add_edge(new_node, edge->get_dst());
            edge->get_dst()->replace_operand(node_name, new_node);
            this->remove_edge(edge);
          }
        }
      }
    }
  }

  // remove all merged nodes.
  for (auto const& node_name : node_list) {
    this->remove_node(node_name);
  }
}

bool Graph::has_node(const std::string& node_name) const {
  return this->nodes.find(node_name) != this->nodes.end();
}

bool Graph::has_edge(const std::string& node_src,
                     const std::string& node_dst) const {
  if (this->edges.count(node_src) == 0) return false;
  if (std::find_if(this->edges.at(node_src).begin(),
                   this->edges.at(node_src).end(),
                   [&](std::shared_ptr<Edge> edge) -> bool {
                     return edge->get_dst()->get_name() == node_dst;
                   }) == this->edges.at(node_src).end())
    return false;
  return true;
}

bool Graph::has_control_edge(const std::string& node_src,
                             const std::string& node_dst) const {
  if (this->control_edges.count(node_src) == 0) return false;
  if (std::find_if(this->control_edges.at(node_src).begin(),
                   this->control_edges.at(node_src).end(),
                   [&](std::shared_ptr<ControlEdge> edge) -> bool {
                     return edge->get_dst()->get_name() == node_dst;
                   }) == this->control_edges.at(node_src).end())
    return false;
  return true;
}

std::shared_ptr<Op> Graph::get_node(const std::string& node) {
  if (this->nodes.find(node) == this->nodes.end()) {
    LOG(FATAL) << "Node " << node << " not in graph";
  }

  return this->nodes[node];
}

std::shared_ptr<const Op> Graph::get_node(const std::string& node) const {
  if (this->nodes.find(node) == this->nodes.end()) {
    LOG(FATAL) << "Node " << node << " not in graph";
  }

  return std::static_pointer_cast<const Op>(this->nodes.at(node));
}

std::vector<std::string> Graph::get_node(
    std::function<bool(const Op*)> pred) const {
  std::vector<std::string> result;

  for (auto const& [node_name, node] : this->nodes) {
    if (pred(node.get())) result.push_back(node_name);
  }

  return result;
}

std::shared_ptr<Edge> Graph::get_edge(const std::string& node_src,
                                      const std::string& node_dst) {
  if (this->edges.find(node_src) == this->edges.end()) {
    LOG(FATAL) << "Cannot find edge from " << node_src << " to node "
               << node_dst;
  }

  for (auto& edge : this->edges[node_src]) {
    if (edge->get_dst_name() == node_dst) return edge;
  }

  LOG(FATAL) << "Cannot find edge from " << node_src << " to node " << node_dst;
}

std::vector<std::string> Graph::get_node_list() const {
  std::vector<std::string> node_list;

  for (auto const& [node_name, node] : this->nodes) {
    node_list.push_back(node_name);
  }

  return node_list;
}

std::vector<std::string> Graph::get_node_list_by_type(
    const OpType& op_type) const {
  std::vector<std::string> node_list;

  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_type() == op_type) {
      node_list.push_back(node_name);
    }
  }

  return node_list;
}

int Graph::get_node_num(OpType op_type) const {
  int cnt = 0;
  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_type() == op_type) ++cnt;
  }

  return cnt;
}

std::vector<std::shared_ptr<Edge>> Graph::get_node_output_edges(
    const std::string& node) {
  return this->edges[node];
}

std::vector<std::shared_ptr<Edge>> const Graph::get_node_output_edges(
    const std::string& node) const {
  return this->edges.at(node);
}

std::vector<std::shared_ptr<ControlEdge>> Graph::get_node_output_control_edges(
    const std::string& node) {
  if (this->control_edges.find(node) == this->control_edges.end()) {
    return std::vector<std::shared_ptr<ControlEdge>>();
  }

  return this->control_edges[node];
}

std::vector<std::shared_ptr<ControlEdge>> const
Graph::get_node_output_control_edges(const std::string& node) const {
  if (this->control_edges.find(node) == this->control_edges.end()) {
    return std::vector<std::shared_ptr<ControlEdge>>();
  }

  return this->control_edges.at(node);
}

std::vector<std::shared_ptr<Edge>> Graph::get_node_input_edges(
    const std::string& node) const {
  std::vector<std::shared_ptr<Edge>> result;

  for (auto const& [node_name, list_edge] : this->edges) {
    for (auto const& edge : list_edge) {
      if (edge->get_dst()->get_name() == node) {
        result.push_back(edge);
      }
    }
  }

  return result;
}

std::vector<std::shared_ptr<ControlEdge>> Graph::get_node_input_control_edges(
    const std::string& node) const {
  std::vector<std::shared_ptr<ControlEdge>> result;

  for (auto const& [node_name, list_edge] : this->control_edges) {
    for (auto const& edge : list_edge) {
      if (edge->get_dst()->get_name() == node) {
        result.push_back(edge);
      }
    }
  }

  return result;
}

void Graph::mark_as_input_node(const std::string& node) {
  if (this->nodes.find(node) == this->nodes.end()) {
    LOG(FATAL) << "Node " << node << " not in graph";
  }

  if (std::find(this->input_nodes.begin(), this->input_nodes.end(), node) !=
      this->input_nodes.end()) {
    return;
  }

  this->input_nodes.push_back(node);
}

void Graph::sort_input_nodes() {
  std::sort(
      this->input_nodes.begin(), this->input_nodes.end(),
      [&](std::string const& node1, std::string const& node2) -> bool {
        return this->get_node(node1)->as<Parameter>()->get_parameter_number() <
               this->get_node(node2)->as<Parameter>()->get_parameter_number();
      });
}

void Graph::align_input_nodes() {
  for (int idx = 0; idx < (int)this->input_nodes.size(); ++idx) {
    this->get_node(this->input_nodes[idx])
        ->as<Parameter>()
        ->set_parameter_number(idx);
  }
}

const std::vector<std::string>& Graph::get_extended_input_nodes() const {
  return this->extended_input_nodes;
}

void Graph::mark_as_extended_input_node(const std::string& node) {
  if (this->nodes.find(node) == this->nodes.end()) {
    LOG(FATAL) << "Node " << node << " not in graph";
  }

  if (std::find(this->input_nodes.begin(), this->input_nodes.end(), node) !=
      this->input_nodes.end()) {
    LOG(FATAL) << "Node " << node
               << " already been marked as input node, cannot be marked as "
                  "extended input node.";
  }

  if (std::find(this->extended_input_nodes.begin(),
                this->extended_input_nodes.end(),
                node) != this->extended_input_nodes.end()) {
    return;
  }

  this->extended_input_nodes.push_back(node);
}

const std::vector<std::string>& Graph::get_input_nodes() const {
  return this->input_nodes;
}

//    void Graph::replace_input_node(std::string old_node_name, std::string
//    new_node_name) {
//        for (auto &node_name : this->input_nodes) {
//            if (node_name == old_node_name) {
//                node_name = new_node_name;
//                return;
//            }
//        }
//
//        LOG(FATAL) << "Cannot replace input node " << old_node_name << " with
//        " << new_node_name << " not in graph";
//    }

std::string Graph::get_input_node(int idx) const {
  if (idx > this->input_nodes.size()) LOG(FATAL) << "Index out of range";

  return this->input_nodes[idx];
}

int Graph::get_input_node_count() const { return this->input_nodes.size(); }

bool Graph::is_input_node(const std::string& node_name) const {
  return std::find(this->input_nodes.begin(), this->input_nodes.end(),
                   node_name) != this->input_nodes.end();
}

void Graph::mark_as_output_node(const std::string& node) {
  if (this->nodes.find(node) == this->nodes.end()) {
    LOG(FATAL) << "Node " << node << " not in graph";
  }

  if (std::find(this->output_nodes.begin(), this->output_nodes.end(), node) !=
      this->output_nodes.end()) {
    return;
  }

  this->output_nodes.push_back(node);
}

const std::vector<std::string>& Graph::get_output_nodes() const {
  return this->output_nodes;
}

std::string Graph::get_output_node(int idx) const {
  return this->output_nodes[idx];
}

int Graph::get_output_node_count() const { return this->output_nodes.size(); }

int Graph::find_output_node_idx(const std::string& node_name) const {
  std::vector<std::string>::const_iterator iter = std::find(
      this->output_nodes.begin(), this->output_nodes.end(), node_name);

  if (iter == this->output_nodes.end()) {
    LOG(FATAL) << "Cannot find output node " << node_name;
  }

  return iter - this->output_nodes.begin();
}

bool Graph::is_output_node(const std::string& node_name) const {
  return std::find(this->output_nodes.begin(), this->output_nodes.end(),
                   node_name) != this->output_nodes.end();
}

void Graph::update_output_node(int idx, const std::string& new_node) {
  EXPECT_TRUE(this->has_node(new_node), "Node " + new_node + " not in graph");
  this->output_nodes[idx] = new_node;
}

void Graph::verify() {
  std::unordered_map<std::string, int> in_degree;

  for (auto const& node : this->nodes) {
    in_degree[node.first] = 0;
  }

  for (auto const& node : this->edges) {
    for (auto const& edge : node.second) {
      std::string dst_name = edge->get_dst()->get_name();
      in_degree[dst_name] = in_degree[dst_name] + 1;
    }
  }

  std::unordered_set<std::string> start_nodes;
  for (auto const& node : this->input_nodes) {
    start_nodes.insert(node);
  }

  for (auto const& node : this->extended_input_nodes) {
    start_nodes.insert(node);
  }

  if (start_nodes.size() == 0) {
    LOG(FATAL) << "Graph must have input nodes";
  }

  for (auto const& node : start_nodes) {
    if (in_degree[node] > 0) {
      LOG(FATAL) << "Node " << node << " have in degree of " << in_degree[node]
                 << " cannot be set as input node";
    }
  }

  std::unordered_set<std::string> visit = this->bfs(start_nodes);

  for (auto const& node : this->output_nodes) {
    if (visit.find(node) == visit.end()) {
      LOG(FATAL) << "Output node " << node << " cannot be reached";
    }
  }

  for (auto const& degree : in_degree) {
    if (degree.second == 0) {
      std::shared_ptr<Op> node = this->get_node(degree.first);
      if (std::find(this->input_nodes.begin(), this->input_nodes.end(),
                    degree.first) == this->input_nodes.end() &&
          std::find(this->extended_input_nodes.begin(),
                    this->extended_input_nodes.end(),
                    degree.first) == this->input_nodes.end()) {
        LOG(FATAL) << "Node " << node->get_name() << " has no valid operand";
      }
    }
  }

  this->verify_is_acyclic();
}

void Graph::set_node_attribute(std::shared_ptr<Op> node, const std::string& key,
                               const std::string& value) {
  node->set_attribute(key, value);
}

void Graph::set_node_attribute(const std::string& node_name,
                               const std::string& key,
                               const std::string& value) {
  this->get_node(node_name)->set_attribute(key, value);
}

std::string Graph::get_node_attribute(std::shared_ptr<Op> node,
                                      const std::string& key) const {
  return node->get_attribute(key);
}

std::string Graph::get_node_attribute(const std::string& node_name,
                                      const std::string& key) const {
  return this->get_node(node_name)->get_attribute(key);
}

bool Graph::node_has_attribute(std::shared_ptr<Op> node,
                               const std::string& key) const {
  return node->has_attribute(key);
}

bool Graph::node_has_attribute(const std::string& node_name,
                               const std::string& key) const {
  return this->get_node(node_name)->has_attribute(key);
}

std::unordered_set<std::string> Graph::bfs(
    const std::unordered_set<std::string>& start_nodes) {
  return this->bfs(start_nodes, [](std::shared_ptr<Op> node) {
    // do nothing
  });
}

std::unordered_set<std::string> Graph::bfs(
    const std::unordered_set<std::string>& start_nodes,
    std::function<void(std::shared_ptr<Op>)> func) {
  std::unordered_set<std::string> visit;

  std::queue<std::string> q;
  for (auto const& node : start_nodes) {
    q.push(node);
  }

  while (!q.empty()) {
    std::string head = q.front();
    q.pop();

    if (visit.find(head) != visit.end()) continue;
    visit.insert(head);

    std::shared_ptr<Op> node_ptr = this->get_node(head);
    func(node_ptr);

    for (auto const& edge : this->edges[head]) {
      q.push(edge->get_dst()->get_name());
    }

    for (auto const& control_edge : this->control_edges[head]) {
      q.push(control_edge->get_dst()->get_name());
    }
  }

  return visit;
}

std::unordered_set<std::string> Graph::post_order(
    const std::string& start_node) {
  return this->post_order(start_node, [](std::shared_ptr<Op> func) {
    // do nothing
  });
}

std::unordered_set<std::string> Graph::post_order(
    const std::string& start_node,
    std::function<void(std::shared_ptr<Op>)> func) {
  std::unordered_set<std::string> visit;

  this->post_order_impl(start_node, func, visit);

  return visit;
}

std::unordered_set<std::string> Graph::post_order_visit_all_nodes(
    std::function<void(std::shared_ptr<Op>)> pre_process,
    std::function<void(std::shared_ptr<Op>)> func) {
  std::vector<std::string> input_nodes = this->get_input_nodes();
  std::unordered_set<std::string> visit;

  for (auto const& input_node : input_nodes) {
    this->post_order_impl(input_node, visit, pre_process, func);
  }

  return visit;
}

std::unordered_set<std::string> Graph::reverse_post_order_visit_all_nodes(
    std::function<void(std::shared_ptr<Op>)> pre_process,
    std::function<void(std::shared_ptr<Op>)> func) {
  std::vector<std::string> output_nodes = this->get_output_nodes();
  std::unordered_set<std::string> visit;

  for (auto const& output_node : output_nodes) {
    this->reverse_post_order_impl(output_node, visit, pre_process, func);
  }

  return visit;
}

std::vector<std::string> Graph::traverse_in_topology_order() const {
  std::unordered_map<std::string, int> steps;

  for (auto const& node_name : this->get_node_list()) {
    steps[node_name] = -1;
  }

  std::queue<std::pair<std::string, int>> q;

  for (auto const& node_name : this->get_input_nodes()) {
    q.push(std::make_pair(node_name, 0));
  }

  for (auto const& node_name : this->get_extended_input_nodes()) {
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

      for (auto const& control_edge :
           this->get_node_output_control_edges(node_name)) {
        q.push(std::make_pair(control_edge->get_dst()->get_name(), step + 1));
      }
    }
  }

  std::vector<std::pair<std::string, int>> traverse_order;

  std::vector<std::string> dead_nodes;

  for (auto const& [node_name, step] : steps) {
    if (step == -1) {
      dead_nodes.push_back(node_name);
    }

    traverse_order.push_back(std::make_pair(node_name, step));
  }

  if (!dead_nodes.empty()) {
    LOG(FATAL) << "Found uneliminated dead node:"
               << mononn_engine::helpers::join(", ", dead_nodes);
  }

  std::sort(traverse_order.begin(), traverse_order.end(),
            [](std::pair<std::string, int> const& a,
               std::pair<std::string, int> const& b) -> bool {
              return a.second < b.second;
            });

  std::vector<std::string> result;
  for (auto const& node_name : traverse_order) {
    result.push_back(node_name.first);
  }

  return result;
}

void Graph::wave_front_order(
    std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func) {
  std::vector<std::string> traverse_order = this->traverse_in_topology_order();

  for (int idx = 0; idx < (int)traverse_order.size(); ++idx) {
    if (idx == (int)traverse_order.size() - 1) {
      func(this->get_node(traverse_order[idx]), nullptr);
    } else {
      func(this->get_node(traverse_order[idx]),
           this->get_node(traverse_order[idx + 1]));
    }
  }
}

void Graph::wave_front_order(
    std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
        func) const {
  std::vector<std::string> traverse_order = this->traverse_in_topology_order();

  for (int idx = 0; idx < (int)traverse_order.size(); ++idx) {
    if (idx == (int)traverse_order.size() - 1) {
      func(this->get_node(traverse_order[idx]), nullptr);
    } else {
      func(this->get_node(traverse_order[idx]),
           this->get_node(traverse_order[idx + 1]));
    }
  }
}

void Graph::topology_order(
    std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func) {
  this->wave_front_order(func);
}

void Graph::topology_order(
    std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
        func) const {
  this->wave_front_order(func);
}

void Graph::reverse_topology_order(
    std::function<void(std::shared_ptr<Op>, std::shared_ptr<Op>)> func) {
  std::vector<std::string> traverse_order = this->traverse_in_topology_order();

  std::reverse(traverse_order.begin(), traverse_order.end());

  for (int idx = 0; idx < (int)traverse_order.size(); ++idx) {
    if (idx == (int)traverse_order.size() - 1) {
      func(this->get_node(traverse_order[idx]), nullptr);
    } else {
      func(this->get_node(traverse_order[idx]),
           this->get_node(traverse_order[idx + 1]));
    }
  }
}

void Graph::reverse_topology_order(
    std::function<void(std::shared_ptr<const Op>, std::shared_ptr<const Op>)>
        func) const {
  std::vector<std::string> traverse_order = this->traverse_in_topology_order();

  std::reverse(traverse_order.begin(), traverse_order.end());

  for (int idx = 0; idx < (int)traverse_order.size(); ++idx) {
    if (idx == (int)traverse_order.size() - 1) {
      func(this->get_node(traverse_order[idx]), nullptr);
    } else {
      func(this->get_node(traverse_order[idx]),
           this->get_node(traverse_order[idx + 1]));
    }
  }
}

std::string Graph::summary() const {
  std::stringstream graph_summary;

  graph_summary << "Graph summary:";

  graph_summary << "Total nodes: " << this->nodes.size() << "\n";

  graph_summary
      << "Total edges: "
      << std::accumulate(
             this->edges.begin(), this->edges.end(), 0,
             [](int total_count,
                const std::pair<std::string,
                                std::vector<std::shared_ptr<Edge>>>& edges)
                 -> int { return total_count + (int)edges.second.size(); })
      << "\n";

  graph_summary << "Input nodes: "
                << std::accumulate(
                       this->input_nodes.begin(), this->input_nodes.end(),
                       std::string(""),
                       [](std::string s, std::string r) -> std::string {
                         return s + " " + r;
                       })
                << "\n";

  graph_summary << "Extended input nodes: "
                << std::accumulate(
                       this->extended_input_nodes.begin(),
                       this->extended_input_nodes.end(), std::string(""),
                       [](std::string s, std::string r) -> std::string {
                         return s + " " + r;
                       })
                << "\n";

  graph_summary << "Output nodes: "
                << std::accumulate(
                       this->output_nodes.begin(), this->output_nodes.end(),
                       std::string(""),
                       [](std::string s, std::string r) -> std::string {
                         return s + " " + r;
                       })
                << "\n";

  std::map<OpType, std::vector<std::string>> op_list_by_type;
  for (auto const& node_name : this->get_node_list()) {
    std::shared_ptr<const Op> node = this->get_node(node_name);
    OpType op_type = node->get_type();

    if (op_list_by_type.find(op_type) == op_list_by_type.end()) {
      op_list_by_type[op_type] = std::vector<std::string>();
    }

    op_list_by_type[op_type].push_back(node_name);
  }

  for (auto const& [op_type, node_list] : op_list_by_type) {
    graph_summary << op_type.to_string() << ": ";

    for (auto const& node_name : node_list) {
      graph_summary << " " << node_name;
    }

    graph_summary << "\n";
  }

  graph_summary << "============ Graph Topology Summary ============\n";

  for (auto const& node_name : this->traverse_in_topology_order()) {
    auto const node = this->get_node(node_name);
    if (node->get_type() == OpType::global_sync) {
      continue;
    }

    graph_summary << node_name << " <- (";
    for (auto const& operand : node->get_operands()) {
      graph_summary << " " << operand->get_name();
    }

    graph_summary << ")\n";
  }

  graph_summary << "============\ Graph Topology Summary End ============\n";

  //        graph_summary << "Grpah topo: " << "\n";
  //        for (auto const &node : this->nodes) {
  //            graph_summary << node.first << ":";
  //
  //            if (this->edges.find(node.first) != this->edges.end()) {
  //                for (auto const &edge : this->edges.at(node.first)) {
  //                    graph_summary << " " << edge->get_dst()->get_name();
  //                }
  //            }
  //
  //            graph_summary << "\n";
  //        }

  return graph_summary.str();
}

std::shared_ptr<Graph> Graph::get_subgraph(
    const std::unordered_set<std::string>& subgraph_node_list,
    const std::unordered_set<std::string>& subgraph_input_nodes,
    const std::unordered_set<std::string>& subgraph_output_nodes) {
  for (auto const& node : subgraph_input_nodes) {
    if (subgraph_node_list.find(node) == subgraph_node_list.end()) {
      LOG(FATAL) << "Input node " << node << " must in node list";
    }
  }

  for (auto const& node : subgraph_output_nodes) {
    if (subgraph_node_list.find(node) == subgraph_node_list.end()) {
      LOG(FATAL) << "Output node " << node << " must in node list";
    }
  }

  std::shared_ptr<Graph> sub_graph = std::make_shared<Graph>();

  for (auto const& node_name : subgraph_node_list) {
    std::shared_ptr<Op> node = this->get_node(node_name);
    sub_graph->add_node(node);
  }

  for (auto const& node_name : subgraph_node_list) {
    for (auto const& edge : this->edges[node_name]) {
      std::string node_dst = edge->get_dst()->get_name();
      if (subgraph_node_list.find(node_dst) != subgraph_node_list.end()) {
        sub_graph->add_edge(edge);
      }
    }
  }

  for (auto const& node : subgraph_input_nodes) {
    sub_graph->mark_as_input_node(node);
  }

  for (auto const& node : subgraph_output_nodes) {
    sub_graph->mark_as_output_node(node);
  }

  return sub_graph;
}

void Graph::post_order_impl(std::string start_node,
                            std::function<void(std::shared_ptr<Op>)> func,
                            std::unordered_set<std::string>& visit) {
  this->post_order_impl(
      start_node, visit, [&](std::shared_ptr<Op> op) -> void {}, func);
}

void Graph::post_order_impl(
    std::string start_node, std::unordered_set<std::string>& visit,
    std::function<void(std::shared_ptr<Op>)> pre_process,
    std::function<void(std::shared_ptr<Op>)> func) {
  if (visit.find(start_node) != visit.end()) return;

  visit.insert(start_node);

  pre_process(this->get_node(start_node));

  for (auto const& edge : this->get_node_output_edges(start_node)) {
    this->post_order_impl(edge->get_src()->get_name(), visit, pre_process,
                          func);
  }

  func(this->get_node(start_node));
}

void Graph::reverse_post_order_impl(
    std::string start_node, std::unordered_set<std::string>& visit,
    std::function<void(std::shared_ptr<Op>)> pre_process,
    std::function<void(std::shared_ptr<Op>)> func) {
  if (visit.find(start_node) != visit.end()) return;

  visit.insert(start_node);

  pre_process(this->get_node(start_node));

  for (auto const& operand : this->get_node(start_node)->get_operands()) {
    this->reverse_post_order_impl(operand->get_name(), visit, pre_process,
                                  func);
  }

  func(this->get_node(start_node));
}

void Graph::clustering() {
  if (!std::all_of(
          this->nodes.begin(), this->nodes.end(),
          [](std::pair<std::string, std::shared_ptr<Op>> const& node) -> bool {
            return node.second->get_cluster_type() == ClusterType::None;
          })) {
    LOG(FATAL) << "Graph already been clustered";
  }

  this->build_transitive_closure();

  this->reduce_cluster_id = 1;
  this->elewise_cluster_id = 1;
  this->gemm_epilogue_cluster_id = 1;
  this->conv_epilogue_cluster_id = 1;
  this->gemm_cluster_id = 1;
  this->conv_cluster_id = 1;

  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_cluster_type() != ClusterType::None) continue;

    if (node->get_type() == OpType::reduce) {
      this->cluster_reduce(node);
      this->reduce_cluster_id += 1;
    }
  }

  for (auto const& [node_name, node] : this->nodes) {
    if (node->is_elewise() && node->get_cluster_type() == ClusterType::None) {
      this->cluster_elewise(node);
      this->elewise_cluster_id += 1;
    }
  }

  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_cluster_type() != ClusterType::None) continue;

    if (node->is_gemm()) {
      this->cluster_gemm(node);
      this->gemm_cluster_id += 1;
    }

    if (node->is_conv()) {
      this->cluster_conv(node);
      this->conv_cluster_id += 1;
    }
  }

  LOG(INFO) << "Epilogue clustering disabled";
  // for (auto const &[node_name, node] : this->nodes) {
  //     if (node->get_cluster_type() != ClusterType::None) continue;

  //     if (node->is_gemm()) {
  //         this->cluster_gemm_epilogue(node);
  //         this->gemm_epilogue_cluster_id += 1;
  //     }

  //     if (node->is_conv()) {
  //         this->cluster_conv_epilogue(node);
  //         this->conv_epilogue_cluster_id += 1;
  //     }
  // }

  LOG(INFO) << "Clustering result:";
  LOG(INFO) << "  Reduce cluster: " << this->reduce_cluster_id - 1;
  LOG(INFO) << "  Elewise cluster: " << this->elewise_cluster_id - 1;

  LOG(INFO) << "  Gemm cluster: " << this->gemm_cluster_id - 1;
  LOG(INFO) << "  Conv cluster: " << this->conv_cluster_id - 1;

  LOG(INFO) << "  Gemm epilogue cluster: "
            << this->gemm_epilogue_cluster_id - 1;
  LOG(INFO) << "  Conv epilogue cluster: "
            << this->conv_epilogue_cluster_id - 1;

  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_cluster_type() == ClusterType::None) {
      LOG(FATAL) << "Node " << node_name << " do not belone to any cluster";
    }
  }
}

bool Graph::is_clustered() const {
  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_cluster_type() == ClusterType::None) {
      return false;
    }
  }

  return true;
}

void Graph::build_transitive_closure(bool including_control_dependency) {
  if (!this->transitive_closure.empty()) {
    this->transitive_closure.clear();
  }

  std::unordered_map<std::string, int> node_id;
  std::vector<std::string> id_to_node_name;
  int cnt = 0;
  for (auto const& [node_name, _] : this->nodes) {
    node_id[node_name] = cnt++;
    id_to_node_name.push_back(node_name);
  }

  std::vector<std::vector<int>> dis(cnt);
  for (int idx = 0; idx < cnt; ++idx) {
    dis[idx] = std::vector<int>(cnt, -1);
  }

  for (auto const& [node_name, edge_list] : this->edges) {
    for (auto const& edge : edge_list) {
      //                this->transitive_closure[std::make_pair(edge->get_src_name(),
      //                edge->get_dst_name())] = 1;
      dis[node_id[edge->get_src_name()]][node_id[edge->get_dst_name()]] = 1;
    }
  }

  if (including_control_dependency) {
    for (auto const& [node_name, edge_list] : this->control_edges) {
      for (auto const& edge : edge_list) {
        //                this->transitive_closure[std::make_pair(edge->get_src_name(),
        //                edge->get_dst_name())] = 1;
        dis[node_id[edge->get_src_name()]][node_id[edge->get_dst_name()]] = 1;
      }
    }
  }

  for (int k = 0; k < cnt; ++k) {
    for (int i = 0; i < cnt; ++i) {
      for (int j = 0; j < cnt; ++j) {
        if (dis[i][k] != -1 && dis[k][j] != -1) {
          if (dis[i][j] == -1) {
            dis[i][j] = dis[i][k] + dis[k][j];
          } else {
            dis[i][j] = std::min(dis[i][j], dis[i][k] + dis[k][j]);
          }
        }
      }
    }
  }

  for (int i = 0; i < cnt; ++i) {
    for (int j = 0; j < cnt; ++j) {
      if (dis[i][j] != -1) {
        this->transitive_closure[std::make_pair(
            id_to_node_name[i], id_to_node_name[j])] = dis[i][j];
      }
    }
  }

  //        for (auto const &k : this->nodes) {
  //            for (auto const &i : this->nodes) {
  //                for (auto const &j : this->nodes) {
  //                    auto i_k = std::make_pair(i.first, k.first);
  //                    auto k_j = std::make_pair(k.first, j.first);
  //                    auto i_j = std::make_pair(i.first, j.first);
  //                    if (this->transitive_closure.find(i_k) !=
  //                    this->transitive_closure.end() &&
  //                        this->transitive_closure.find(k_j) !=
  //                        this->transitive_closure.end()) {
  //
  //                        if (this->transitive_closure.find(i_j) ==
  //                        this->transitive_closure.end()) {
  //                            this->transitive_closure[i_j] =
  //                            this->transitive_closure[i_k] +
  //                            this->transitive_closure[k_j];
  //                        } else {
  //                            this->transitive_closure[i_j] =
  //                            std::min(this->transitive_closure[i_k] +
  //                            this->transitive_closure[k_j],
  //                                                                     this->transitive_closure[i_j]);
  //                        }
  //                    }
  //                }
  //            }
  //        }
}

void Graph::trace_index(const std::string& index, const std::string& node_name,
                        std::string inverse_reduce_dimension) {
  IndexTracer index_tracer(index);

  // only trace index for nodes with identical sub cluster tag;
  std::string sub_cluster_tag;
  if (this->node_has_attribute(node_name, OpAttribute::sub_cluster_tag)) {
    sub_cluster_tag =
        this->get_node(node_name)->get_attribute(OpAttribute::sub_cluster_tag);
  }

  std::function<void(std::string, std::string)> trace_index_dfs =
      [&](std::string node_name, std::string traced_by) -> void {
    if (this->node_has_attribute(node_name, OpAttribute::sub_cluster_tag) &&
        this->get_node_attribute(node_name, OpAttribute::sub_cluster_tag) !=
            sub_cluster_tag) {
      return;
    }

    if (this->symbolic_index.find(node_name) == this->symbolic_index.end()) {
      this->symbolic_index[node_name] = std::vector<SymbolicIndexStamp>();
    }

    std::shared_ptr<Op> node = this->get_node(node_name);
    std::string index_before_trace = index_tracer.get_index();
    std::string pred_before_trace = index_tracer.get_predictive();

    if (node->get_type() == OpType::gather) {
      index_tracer.trace_gather_operand(std::static_pointer_cast<Gather>(node));
      std::string index_after_trace = index_tracer.get_index();
      std::string pred_after_trace = index_tracer.get_predictive();

      LOG_ONCE(WARNING, __already_logged_gather,
               "May need take care of predictive propagation for gather op");

      // already traced
      if (std::find_if(
              this->symbolic_index[node_name].begin(),
              this->symbolic_index[node_name].end(),
              [&](SymbolicIndexStamp const& index_trace_stamp) -> bool {
                return index_trace_stamp.index_before_trace ==
                           index_before_trace &&
                       index_trace_stamp.index_after_trace == index_after_trace;
              }) != this->symbolic_index[node_name].end()) {
        return;
      }

      SymbolicIndexStamp index_trace_stamp{index_before_trace,
                                           index_after_trace, traced_by,
                                           pred_before_trace, pred_after_trace};

      this->symbolic_index[node_name].push_back(index_trace_stamp);

      trace_index_dfs(node->get_operand(0)->get_name(), node_name);

      index_tracer.set_index(index_before_trace);
      index_tracer.set_pred(pred_before_trace);
      index_tracer.trace_gather_indices(std::static_pointer_cast<Gather>(node));
      trace_index_dfs(node->get_operand(1)->get_name(), node_name);

      return;
    }

    if (node->get_type() == OpType::reduce) {
      index_tracer.trace_reduce(std::static_pointer_cast<Reduce>(node),
                                inverse_reduce_dimension);
      std::string index_after_trace = index_tracer.get_index();
      std::string pred_after_trace = index_tracer.get_predictive();

      // already traced
      if (std::find_if(
              this->symbolic_index[node_name].begin(),
              this->symbolic_index[node_name].end(),
              [&](SymbolicIndexStamp const& index_trace_stamp) -> bool {
                return index_trace_stamp.index_before_trace ==
                           index_before_trace &&
                       index_trace_stamp.index_after_trace == index_after_trace;
              }) != this->symbolic_index[node_name].end()) {
        return;
      }

      SymbolicIndexStamp index_trace_stamp{index_before_trace,
                                           index_after_trace, traced_by,
                                           pred_before_trace, pred_after_trace};

      this->symbolic_index[node_name].push_back(index_trace_stamp);

      trace_index_dfs(node->get_operand(0)->get_name(), node_name);

      index_tracer.set_index(index_after_trace);
      index_tracer.set_pred(pred_after_trace);
      trace_index_dfs(node->get_operand(1)->get_name(), node_name);

      return;
    }

    if (node->get_type() == OpType::concatenate) {
      // already traced
      if (std::find_if(
              this->symbolic_index[node_name].begin(),
              this->symbolic_index[node_name].end(),
              [&](SymbolicIndexStamp const& index_trace_stamp) -> bool {
                return index_trace_stamp.index_before_trace ==
                       index_before_trace;
              }) != this->symbolic_index[node_name].end()) {
        return;
      }

      SymbolicIndexStamp index_trace_stamp{
          index_before_trace,
          index_before_trace /*Use same after trace index for concat node*/,
          traced_by, pred_before_trace,
          pred_before_trace /*Use save after trace pred*/};
      this->symbolic_index[node_name].push_back(index_trace_stamp);

      for (int idx = 0; idx < node->get_operand_count(); ++idx) {
        index_tracer.set_index(index_before_trace);
        index_tracer.set_pred(pred_before_trace);

        index_tracer.trace_concatenate(
            std::static_pointer_cast<Concatenate>(node), idx);

        auto const& operand = node->get_operand(idx);
        trace_index_dfs(operand->get_name(), node_name);
      }

      return;
    }

    if (node->get_type() == OpType::dynamic_update_slice) {
      // already traced
      if (std::find_if(
              this->symbolic_index[node_name].begin(),
              this->symbolic_index[node_name].end(),
              [&](SymbolicIndexStamp const& index_trace_stamp) -> bool {
                return index_trace_stamp.index_before_trace ==
                       index_before_trace;
              }) != this->symbolic_index[node_name].end()) {
        return;
      }

      index_tracer.trace_dynamic_update_slice_operand(
          std::static_pointer_cast<DynamicUpdateSlice>(node));

      SymbolicIndexStamp index_trace_stamp_operand{
          index_before_trace, index_tracer.get_index(), traced_by,
          pred_before_trace, index_tracer.get_predictive()};
      this->symbolic_index[node_name].push_back(index_trace_stamp_operand);

      trace_index_dfs(node->get_operand(0)->get_name(), node_name);

      index_tracer.set_index(index_before_trace);
      index_tracer.set_pred(pred_before_trace);

      index_tracer.trace_dynamic_update_slice_update(
          std::static_pointer_cast<DynamicUpdateSlice>(node));
      trace_index_dfs(node->get_operand(1)->get_name(), node_name);

      for (int idx = 2; idx < node->get_operand_count(); ++idx) {
        index_tracer.set_index(index_before_trace);
        index_tracer.set_pred(pred_before_trace);
        trace_index_dfs(node->get_operand(idx)->get_name(), node_name);
      }

      return;
    }

    index_tracer.trace(node);
    std::string index_after_trace = index_tracer.get_index();
    std::string pred_after_trace = index_tracer.get_predictive();

    // already traced
    if (std::find_if(this->symbolic_index[node_name].begin(),
                     this->symbolic_index[node_name].end(),
                     [&](SymbolicIndexStamp const& index_trace_stamp) -> bool {
                       return index_trace_stamp.index_before_trace ==
                                  index_before_trace &&
                              index_trace_stamp.index_after_trace ==
                                  index_after_trace;
                     }) != this->symbolic_index[node_name].end()) {
      return;
    }

    SymbolicIndexStamp index_trace_stamp{index_before_trace, index_after_trace,
                                         traced_by, pred_before_trace,
                                         pred_after_trace};
    this->symbolic_index[node_name].push_back(index_trace_stamp);

    for (int idx = 0; idx < node->get_operand_count(); ++idx) {
      auto const& operand = node->get_operand(idx);

      index_tracer.set_index(index_after_trace);
      index_tracer.set_pred(pred_after_trace);

      if (node->get_type() == OpType::dynamic_slice && idx >= 1) {
        index_tracer.set_index("0");
        index_tracer.set_pred("true");
      }

      if (node->get_type() == OpType::reduce_window &&
          idx >= node->get_operand_count() / 2) {
        index_tracer.set_index("0");
        index_tracer.set_pred("true");
      }

      trace_index_dfs(operand->get_name(), node_name);
    }
  };

  trace_index_dfs(node_name, "");

  for (auto const& [traced_node_name, traced_index_list] :
       this->symbolic_index) {
    // Theoratically any node can be traced by multiple index.
    if (traced_index_list.size() > 1) {
      //     EXPECT_TRUE(this->get_node(traced_node_name)->get_type() ==
      //     OpType::constant ||
      //         this->get_node(traced_node_name)->get_type() ==
      //         OpType::parameter, "Node " + traced_node_name + " have multiple
      //         traced index however it's neither constant nor parameter");

      if (this->get_node(traced_node_name)->get_type() == OpType::parameter) {
        for (auto const& its : traced_index_list) {
          std::shared_ptr<Op> next_node = this->get_node(its.traced_by);
          //             EXPECT_TRUE(next_node->get_type() == OpType::slice,
          //                 "Node " + traced_node_name + " have multiple index
          //                 but it's upstream traced node " + its.traced_by + "
          //                 is not slice");
          if (next_node->get_type() == OpType::slice) {
            next_node->get_implementation()->add_operand_reuse_mask(
                traced_node_name,
                traced_node_name + "_reuse_" + next_node->get_name());
          }
        }
      }
    }

    this->get_node(traced_node_name)->set_symbolic_index(traced_index_list);
    // this->get_node(traced_node_name)->propagate_index_to_implementation();
  }
}

//    void Graph::trace_ilp_index(int ilp_id, const std::string &index, const
//    std::string &node_name, std::string inverse_reduce_dimension) {
//        IndexTracer index_tracer(index);
//
//        // only trace node index have same sub cluster tag;
//        std::string sub_cluster_tag;
//        if (this->node_has_attribute(node_name, OpAttribute::sub_cluster_tag))
//        {
//            sub_cluster_tag =
//            this->get_node(node_name)->get_attribute(OpAttribute::sub_cluster_tag);
//        }
//
//        std::function<void(std::string, std::string)> trace_index_dfs =
//        [&](std::string node_name, std::string traced_by) -> void {
//            if (this->node_has_attribute(node_name,
//            OpAttribute::sub_cluster_tag) &&
//                this->get_node_attribute(node_name,
//                OpAttribute::sub_cluster_tag) != sub_cluster_tag) { return;
//            }
//
//            if (this->ilp_traced_index.find(node_name) ==
//            this->ilp_traced_index.end()) {
//                this->ilp_traced_index[node_name] =
//                std::vector<std::vector<IndexTraceStamp>>(this->ilp_factor);
//            }
//
//            std::shared_ptr<Op> node = this->get_node(node_name);
//            std::string index_before_trace = index_tracer.get_index();
//            std::string pred_before_trace = index_tracer.get_predictive();
//
//            if (node->get_type() == OpType::gather) {
//                LOG_ONCE(WARNING, __already_logged__gather, "May need take
//                care of predictive propagation for gather op");
//                index_tracer.trace_gather_operand_ilp(std::static_pointer_cast<Gather>(node),
//                ilp_id); std::string index_after_trace =
//                index_tracer.get_index(); std::string pred_after_trace =
//                index_tracer.get_predictive();
//
//                // already traced
//                if
//                (std::find_if(this->ilp_traced_index[node_name][ilp_id].begin(),
//                this->ilp_traced_index[node_name][ilp_id].end(),
//                [&](IndexTraceStamp const &index_trace_stamp) -> bool {
//                    return index_trace_stamp.index_before_trace ==
//                    index_before_trace && index_trace_stamp.index_after_trace
//                    == index_after_trace;
//                }) != this->ilp_traced_index[node_name][ilp_id].end()) {
//                    return;
//                }
//
//                IndexTraceStamp index_trace_stamp { index_before_trace,
//                index_after_trace, traced_by, pred_before_trace,
//                pred_after_trace};
//
//                this->ilp_traced_index[node_name][ilp_id].push_back(index_trace_stamp);
//
//                trace_index_dfs(node->get_operand(0)->get_name(), node_name);
//
//                index_tracer.set_index(index_before_trace);
//                index_tracer.set_pred(pred_before_trace);
//                index_tracer.trace_gather_indices(std::static_pointer_cast<Gather>(node));
//                trace_index_dfs(node->get_operand(1)->get_name(), node_name);
//
//                return;
//            }
//
//            if (node->get_type() == OpType::reduce) {
//                index_tracer.trace_reduce(std::static_pointer_cast<Reduce>(node),
//                inverse_reduce_dimension); std::string index_after_trace =
//                index_tracer.get_index(); std::string pred_after_trace =
//                index_tracer.get_predictive();
//
//                // already traced
//                if
//                (std::find_if(this->ilp_traced_index[node_name][ilp_id].begin(),
//                this->ilp_traced_index[node_name][ilp_id].end(),
//                [&](IndexTraceStamp const &index_trace_stamp) -> bool {
//                    return index_trace_stamp.index_before_trace ==
//                    index_before_trace && index_trace_stamp.index_after_trace
//                    == index_after_trace;
//                }) != this->ilp_traced_index[node_name][ilp_id].end()) {
//                    return;
//                }
//
//                IndexTraceStamp index_trace_stamp { index_before_trace,
//                index_after_trace, traced_by, pred_before_trace,
//                pred_after_trace};
//
//                this->ilp_traced_index[node_name][ilp_id].push_back(index_trace_stamp);
//
//                trace_index_dfs(node->get_operand(0)->get_name(), node_name);
//
//                index_tracer.set_index(index_after_trace);
//                index_tracer.set_pred(pred_after_trace);
//                trace_index_dfs(node->get_operand(1)->get_name(), node_name);
//
//                return;
//            }
//
//            if (node->get_type() == OpType::concatenate) {
//                // already traced
//                if
//                (std::find_if(this->ilp_traced_index[node_name][ilp_id].begin(),
//                this->ilp_traced_index[node_name][ilp_id].end(),
//                [&](IndexTraceStamp const &index_trace_stamp) -> bool {
//                    return index_trace_stamp.index_before_trace ==
//                    index_before_trace;
//                }) != this->ilp_traced_index[node_name][ilp_id].end()) {
//                    return;
//                }
//
//                IndexTraceStamp index_trace_stamp { index_before_trace,
//                index_before_trace /*Use same after trace index for concat
//                node*/, traced_by, pred_before_trace, pred_before_trace /*Use
//                save after trace pred*/ };
//                this->ilp_traced_index[node_name][ilp_id].push_back(index_trace_stamp);
//
//                for (int idx = 0; idx < node->get_operand_count(); ++idx) {
//                    index_tracer.set_index(index_before_trace);
//                    index_tracer.set_pred(pred_before_trace);
//
//                    index_tracer.trace_concatenate(std::static_pointer_cast<Concatenate>(node),
//                    idx);
//
//                    auto const &operand = node->get_operand(idx);
//                    trace_index_dfs(operand->get_name(), node_name);
//                }
//
//                return;
//            }
//
//            index_tracer.trace(node);
//            std::string index_after_trace = index_tracer.get_index();
//            std::string pred_after_trace = index_tracer.get_predictive();
//
//            // already traced
//            if
//            (std::find_if(this->ilp_traced_index[node_name][ilp_id].begin(),
//            this->ilp_traced_index[node_name][ilp_id].end(),
//            [&](IndexTraceStamp const &index_trace_stamp) -> bool {
//                return index_trace_stamp.index_before_trace ==
//                index_before_trace && index_trace_stamp.index_after_trace ==
//                index_after_trace;
//            }) != this->ilp_traced_index[node_name][ilp_id].end()) {
//                return;
//            }
//
//            IndexTraceStamp index_trace_stamp { index_before_trace,
//            index_after_trace, traced_by, pred_before_trace, pred_after_trace
//            };
//            this->ilp_traced_index[node_name][ilp_id].push_back(index_trace_stamp);
//
//            for (int idx = 0; idx < node->get_operand_count(); ++idx) {
//                auto const &operand = node->get_operand(idx);
//                index_tracer.set_index(index_after_trace);
//                index_tracer.set_pred(pred_after_trace);
//                trace_index_dfs(operand->get_name(), node_name);
//            }
//        };
//
//        trace_index_dfs(node_name, "");
//
//        for (auto const &[traced_node_name, traced_index_list_for_each_ilp_id]
//        : this->ilp_traced_index) {
//            auto const &traced_index_list =
//            traced_index_list_for_each_ilp_id[ilp_id];
//
//            if (traced_index_list.size() > 1) {
//                EXPECT_TRUE(this->get_node(traced_node_name)->get_type() ==
//                OpType::constant ||
//                            this->get_node(traced_node_name)->get_type() ==
//                            OpType::parameter, "Node " + traced_node_name + "
//                            have multiple traced index however it's neither
//                            constant nor parameter");
//
//                for (auto const &its : traced_index_list) {
//                    std::shared_ptr<Op> next_node =
//                    this->get_node(its.traced_by);
////                    EXPECT_TRUE(next_node->get_type() == OpType::slice,
////                                "Node " + traced_node_name + " have multiple
/// index but it's upstream traced node " + its.traced_by + " is not slice");
//
//                    static bool __already_logged = false;
//                    if (!__already_logged) {
//                        __already_logged = true;
//                        LOG(WARNING) << "Operand reuse mask may need further
//                        refactor";
//                    }
//
//                    next_node->get_implementation()->add_operand_reuse_mask(traced_node_name,
//                    traced_node_name + "_reuse_" + next_node->get_name());
//                }
//            }
//
//            this->get_node(traced_node_name)->set_ilp_traced_index(ilp_id,
//            traced_index_list);
//            this->get_node(traced_node_name)->propagate_ilp_index_to_implementation();
//        }
//    }

const std::vector<SymbolicIndexStamp>& Graph::get_symbolic_index(
    const std::string& node_name) const {
  EXPECT_TRUE(this->symbolic_index.count(node_name) > 0,
              "Index " + node_name + " not traced");

  return this->symbolic_index.at(node_name);
}

bool Graph::is_node_traced(const std::string& node_name) const {
  return this->symbolic_index.find(node_name) != this->symbolic_index.end() &&
         this->symbolic_index.at(node_name).size() != 0;
}

void Graph::reset_symbolic_index() { this->symbolic_index.clear(); }

std::vector<std::string> Graph::get_symbolic_index_before_trace(
    const std::string& node_name) const {
  if (this->symbolic_index.find(node_name) == this->symbolic_index.end()) {
    LOG(FATAL) << "Node " << node_name << " do not have traced index";
  }

  if (this->symbolic_index.at(node_name).size() == 0) {
    LOG(FATAL) << "Node " << node_name << " do not have traced index";
  }

  std::vector<std::string> index_list;

  for (auto const& index : this->symbolic_index.at(node_name)) {
    index_list.push_back(index.index_before_trace);
  }

  return index_list;
}

std::vector<std::string> Graph::get_symbolic_index_after_trace(
    const std::string& node_name) const {
  if (this->symbolic_index.find(node_name) == this->symbolic_index.end()) {
    LOG(FATAL) << "Node " << node_name << " do not have traced index";
  }

  if (this->symbolic_index.at(node_name).size() == 0) {
    LOG(FATAL) << "Node " << node_name << " do not have traced index";
  }

  std::vector<std::string> index_list;

  for (auto const& index : this->symbolic_index.at(node_name)) {
    index_list.push_back(index.index_after_trace);
  }

  return index_list;
}

bool Graph::is_simple_padding_scenario() const {
  EXPECT_TRUE(!this->transitive_closure.empty(),
              "Should build transitive closure first");
  EXPECT_TRUE(!this->symbolic_index.empty(), "should trace index first");

  std::vector<std::string> pad_list;
  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_type() == OpType::pad) {
      pad_list.push_back(node_name);
    }
  }

  if (pad_list.empty()) return false;

  for (auto const& node_name : pad_list) {
    // Pad node must be output
    if (std::find(this->output_nodes.begin(), this->output_nodes.end(),
                  node_name) == this->output_nodes.end()) {
      return false;
    }

    // Pad must be traced once
    if (this->get_symbolic_index_before_trace(node_name).size() != 1) {
      return false;
    }
  }

  // Multiple pad must have same parameters
  if (pad_list.size() > 1) {
    for (int idx = 1; idx < pad_list.size(); ++idx) {
      if (this->symbolic_index.at(pad_list[idx]) !=
          this->symbolic_index.at(pad_list[0]))
        return false;
      std::shared_ptr<const Pad> pad1 =
          std::static_pointer_cast<const Pad>(this->get_node(pad_list[0]));
      std::shared_ptr<const Pad> pad2 =
          std::static_pointer_cast<const Pad>(this->get_node(pad_list[idx]));

      if (pad1->get_padding_low() != pad2->get_padding_low()) return false;
      if (pad1->get_padding_high() != pad2->get_padding_high()) return false;
    }
  }

  // Any node in graph must before at least one pad node
  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_type() != OpType::pad) {
      bool before_any_pad_node = false;

      for (auto const& pad_node : pad_list) {
        if (this->topology_before(node_name, pad_node)) {
          before_any_pad_node = true;
          break;
        }
      }

      if (!before_any_pad_node) return false;
    }
  }

  return true;
}

bool Graph::is_dead_node(const std::string& node_name) const {
  EXPECT_TRUE(!this->transitive_closure.empty(),
              "Transitive closure not built yet");

  if (this->get_node(node_name)->get_type() == OpType::global_sync)
    return false;

  for (auto const& output_node_name : this->get_output_nodes()) {
    if (node_name == output_node_name ||
        this->topology_before(node_name, output_node_name))
      return false;
  }

  return true;
}

bool Graph::remain_acyclic_after_node_merge(
    const std::vector<std::string>& node_list) const {
  std::unordered_set<std::string> visit;

  std::function<bool(std::string)> dfs_search =
      [&](std::string node_name) -> bool {
    if (visit.count(node_name)) {
      return true;
    }

    visit.insert(node_name);

    for (auto const& edge : this->get_node_output_edges(node_name)) {
      std::string dst_name = edge->get_dst()->get_name();

      if (std::find(node_list.begin(), node_list.end(), dst_name) !=
          node_list.end()) {
        if (std::find(node_list.begin(), node_list.end(), node_name) !=
            node_list.end()) {
          continue;
        } else {
          return false;
        }
      }

      if (!dfs_search(dst_name)) {
        return false;
      }
    }

    return true;
  };

  for (auto const& node_name : node_list) {
    if (!dfs_search(node_name)) return false;
  }

  return true;
}

void Graph::set_instruction_parallel_factor(int _ilp_factor) {
  this->ilp_factor = _ilp_factor;

  for (auto& [node_name, node] : this->nodes) {
    node->set_instruction_parallel_factor(_ilp_factor);
  }
}

int Graph::get_instruction_parallel_factor() const { return this->ilp_factor; }

bool Graph::is_acyclic(std::vector<std::string>& nodes_in_cycle,
                       bool include_control_edge) const {
  std::vector<std::string> node_list;

  std::unordered_set<std::string> visit;
  std::unordered_set<std::string> visit_in_path;

  std::function<bool(std::string node_name)> dfs =
      [&](std::string node_name) -> bool {
    if (visit_in_path.count(node_name) != 0) {
      node_list.push_back(node_name);
      return false;
    }

    if (visit.count(node_name)) return true;

    node_list.push_back(node_name);
    visit.insert(node_name);
    visit_in_path.insert(node_name);

    for (auto const& edge : this->edges.at(node_name)) {
      if (!dfs(edge->get_dst_name())) return false;
    }

    if (include_control_edge && this->control_edges.count(node_name) > 0) {
      for (auto const& control_edge : this->control_edges.at(node_name)) {
        if (!dfs(control_edge->get_dst_name())) return false;
      }
    }

    node_list.pop_back();
    visit_in_path.erase(node_name);
    return true;
  };

  for (auto const& [node_name, _] : this->nodes) {
    EXPECT_TRUE(visit_in_path.empty(), "");
    EXPECT_TRUE(node_list.empty(), "");

    if (!dfs(node_name)) {
      nodes_in_cycle = node_list;
      return false;
    }
  }

  return true;
}

void Graph::verify_is_acyclic(bool include_control_edge) const {
  std::vector<std::string> node_in_cycle;

  if (!this->is_acyclic(node_in_cycle, true)) {
    LOG(FATAL) << "Acyclic check failed, detected cycle in graph: "
               << mononn_engine::helpers::join("->", node_in_cycle);
  }
}

// std::unique_ptr<Graph> Graph::deep_copy() const {
//     std::unique_ptr<Graph> new_graph = std::make_unique<Graph>();

//     for (auto const& [node_name, node] : this->nodes) {
//         // Convert to shared ptr as a interim solution.
//         new_graph->nodes[node_name] =
//         std::shared_ptr<Op>(node->deep_copy().release());
//     }

//     for (auto const &[src_node_name, edges] : this->edges) {
//         for (auto const& edge : edges) {
//             std::shared_ptr<Edge> new_edge =
//             std::make_shared<Edge>(edge->get_src(), edge->get_dst());
//             new_edge->set_sync(edge->get_sync());
//             new_graph->add_edge(new_edge);
//         }
//     }

//     for (auto const &[src_node_name, control_edges] : this->control_edges) {
//         for (auto const &control_edge : control_edges) {
//             new_graph->add_control_edge(control_edge->get_src_name(),
//             control_edge->get_dst_name());
//         }
//     }

//     new_graph->symbolic_index = this->symbolic_index;

//     new_graph->input_nodes = this->input_nodes;
//     new_graph->extended_input_nodes = this->extended_input_nodes;
//     new_graph->output_nodes = this->output_nodes;
//     new_graph->reduce_cluster_id = this->reduce_cluster_id;
//     new_graph->elewise_cluster_id = this->elewise_cluster_id;
//     new_graph->gemm_epilogue_cluster_id = this->gemm_epilogue_cluster_id;
//     new_graph->conv_epilogue_cluster_id = this->conv_epilogue_cluster_id;
//     new_graph->gemm_cluster_id = this->gemm_cluster_id;
//     new_graph->conv_cluster_id = this->conv_cluster_id;
//     new_graph->ilp_factor = this->ilp_factor;
//     new_graph->transitive_closure = this->transitive_closure;

//     return std::move(new_graph);
// }

bool Graph::topology_before(const std::string& node1,
                            const std::string& node2) const {
  if (node1 == node2) return false;

  if (!this->nodes.count(node1)) {
    LOG(FATAL) << node1 << " not found in graph";
  }

  if (!this->nodes.count(node2)) {
    LOG(FATAL) << node2 << " not found in graph";
  }

  if (this->transitive_closure.find(std::make_pair(node1, node2)) !=
      this->transitive_closure.end())
    return true;
  else
    return false;
}

int Graph::distance(const std::string& node_src,
                    const std::string& node_dst) const {
  if (this->transitive_closure.count(std::make_pair(node_src, node_dst)) == 0) {
    LOG(FATAL) << "No path between " << node_src << " and " << node_dst << ".";
  }

  return this->transitive_closure.at(std::make_pair(node_src, node_dst));
}

bool Graph::reachable_under_constrain(const std::string& node_src,
                                      const std::string& node_dst,
                                      const OpType& op_type_constrain) const {
  if (!this->topology_before(node_src, node_dst)) return false;

  std::unordered_set<std::string> visit;

  std::function<bool(std::string)> dfs_search =
      [&](std::string node_name) -> bool {
    if (node_name == node_dst) return true;
    if (visit.count(node_name) != 0) return false;
    visit.insert(node_name);

    for (auto const& edge : this->get_node_output_edges(node_name)) {
      if (edge->get_dst()->get_type() != op_type_constrain) continue;
      if (dfs_search(edge->get_dst()->get_name())) return true;
    }

    return false;
  };

  return dfs_search(node_src);
}

std::vector<std::string> Graph::search_downstream_node_under_constrain(
    const std::string& node_src, std::function<bool(const Op*)> path_constrain,
    std::function<bool(const Op*, std::vector<std::string> const&)>
        dst_constrain,
    bool return_nodes_in_path) const {
  std::unordered_set<std::string> visit;
  std::vector<std::string> result;

  std::vector<std::string> nodes_in_path;

  std::function<bool(std::string)> dfs_search =
      [&](std::string node_name) -> bool {
    if (visit.count(node_name)) return false;
    visit.insert(node_name);

    for (auto const& edge : this->get_node_output_edges(node_name)) {
      if (dst_constrain(edge->get_dst().get(), nodes_in_path)) {
        result.push_back(node_src);

        if (return_nodes_in_path)
          result.insert(result.end(), nodes_in_path.begin(),
                        nodes_in_path.end());

        result.push_back(edge->get_dst()->get_name());
        return true;
      }

      if (path_constrain(edge->get_dst().get())) {
        nodes_in_path.push_back(edge->get_dst()->get_name());
        if (dfs_search(edge->get_dst()->get_name())) return true;
        nodes_in_path.pop_back();
      }
    }

    return false;
  };

  dfs_search(node_src);
  return result;
}

std::vector<std::string> Graph::search_downstream_node_under_constrain(
    const std::string& node_src,
    std::function<bool(const Op*, std::vector<std::string> const&)>
        dst_constrain,
    bool return_nodes_in_path) const {
  return this->search_downstream_node_under_constrain(
      node_src, [&](const Op* op) -> bool { return true; }, dst_constrain,
      return_nodes_in_path);
}

std::vector<std::string> Graph::search_nodes_between_two_nodes(
    const std::string& node_src, const std::string& node_dst) const {
  std::vector<std::string> result;
  for (auto const& node_name : this->get_node_list()) {
    std::vector<std::shared_ptr<Edge>> const& input_edges =
        this->edges.at(node_src);

    if (std::find_if(input_edges.begin(), input_edges.end(),
                     [&](std::shared_ptr<Edge> const edge) -> bool {
                       return edge->get_dst()->get_name() == node_name;
                     }) != input_edges.end()) {
      std::vector<std::shared_ptr<Edge>> const& output_edges =
          this->edges.at(node_name);

      if (std::find_if(output_edges.begin(), output_edges.end(),
                       [&](std::shared_ptr<Edge> const edge) -> bool {
                         return edge->get_dst()->get_name() == node_dst;
                       }) != output_edges.end()) {
        result.push_back(node_name);
      }
    }
  }

  return result;
}

void Graph::cluster_reduce(std::shared_ptr<Op> node) {
  std::function<void(std::shared_ptr<Op>)> cluster_epilogue =
      [&](std::shared_ptr<Op> n) -> void {
    if (n->get_type() == OpType::broadcast) return;
    if (n->get_cluster_type() != ClusterType::None) return;
    if (!n->is_elewise()) return;

    if (!this->remain_acyclic_after_add_node_to_cluster(
            n, ClusterType::Reduce, this->reduce_cluster_id))
      return;

    n->set_cluster_type(ClusterType::Reduce);
    n->set_cluster_id(this->reduce_cluster_id);

    for (auto const& edge : this->edges[n->get_name()]) {
      cluster_epilogue(edge->get_dst());
    }
  };

  std::function<void(std::shared_ptr<Op>)> cluster_prologue =
      [&](std::shared_ptr<Op> n) -> void {
    if (n->get_cluster_type() != ClusterType::None) return;
    if (this->transitive_closure.find(
            std::make_pair(node->get_name(), n->get_name())) !=
        this->transitive_closure.end()) {
      return;  // prologue most not logical behind reduce op;
    }
    if (!n->is_elewise()) return;

    if (!this->remain_acyclic_after_add_node_to_cluster(
            n, ClusterType::Reduce, this->reduce_cluster_id))
      return;

    n->set_cluster_type(ClusterType::Reduce);
    n->set_cluster_id(this->reduce_cluster_id);

    if (n->get_type() == OpType::broadcast) {
      return;
    }

    for (auto const& operand : n->get_operands()) {
      cluster_prologue(operand);
    }

    for (auto const& edge : this->get_node_output_edges(n->get_name())) {
      cluster_prologue(edge->get_dst());
    }
  };

  node->set_cluster_type(ClusterType::Reduce);
  node->set_cluster_id(this->reduce_cluster_id);

  for (auto const& edge : this->edges[node->get_name()]) {
    cluster_epilogue(edge->get_dst());
  }

  for (auto const& operand : node->get_operands()) {
    cluster_prologue(operand);
  }
}

void Graph::cluster_elewise(std::shared_ptr<Op> node) {
  std::function<void(std::shared_ptr<Op>)> cluster =
      [&](std::shared_ptr<Op> n) -> void {
    if (n->get_cluster_type() != ClusterType::None) return;
    if (!n->is_elewise()) return;
    if (!this->remain_acyclic_after_add_node_to_cluster(
            n, ClusterType::Elewise, this->elewise_cluster_id))
      return;

    n->set_cluster_type(ClusterType::Elewise);
    n->set_cluster_id(this->elewise_cluster_id);

    for (auto const& edge : this->edges[n->get_name()]) {
      cluster(edge->get_dst());
    }

    for (auto const& operand : n->get_operands()) {
      cluster(operand);
    }
  };

  cluster(node);
}

void Graph::cluster_gemm_epilogue(std::shared_ptr<Op> node) {
  std::function<bool(std::shared_ptr<Op>, ClusterType, int)> can_cluster =
      [&](std::shared_ptr<Op> n, ClusterType cluster_type,
          int cluster_id) -> bool {
    if (n->get_type() == OpType::custom_call) return true;

    if (n->get_cluster_type() != cluster_type) return false;
    if (n->get_cluster_id() != cluster_id) return false;

    for (auto const& edge : this->edges[n->get_name()]) {
      if (can_cluster(edge->get_dst(), cluster_type, cluster_id)) return true;
    }

    return false;
  };

  std::function<void(std::shared_ptr<Op>, ClusterType, int)> do_cluster =
      [&](std::shared_ptr<Op> n, ClusterType cluster_type,
          int cluster_id) -> void {
    if (n->get_cluster_type() != cluster_type) return;
    if (n->get_cluster_id() != cluster_id) return;

    n->set_cluster_type(ClusterType::GemmEpilogue);
    n->set_cluster_id(this->gemm_epilogue_cluster_id);

    for (auto const& edge : this->edges[n->get_name()]) {
      do_cluster(edge->get_dst(), cluster_type, cluster_id);
    }

    for (auto const& operand : n->get_operands()) {
      do_cluster(operand, cluster_type, cluster_id);
    }
  };

  node->set_cluster_type(ClusterType::GemmEpilogue);
  node->set_cluster_id(this->gemm_epilogue_cluster_id);

  if (this->edges[node->get_name()].size() != 1) return;

  std::shared_ptr<Op> epilogue_node =
      this->edges[node->get_name()][0]->get_dst();

  if (epilogue_node->get_cluster_type() != ClusterType::Elewise) return;

  if (!can_cluster(epilogue_node, epilogue_node->get_cluster_type(),
                   epilogue_node->get_cluster_id()))
    return;

  do_cluster(epilogue_node, epilogue_node->get_cluster_type(),
             epilogue_node->get_cluster_id());
}

void Graph::cluster_conv_epilogue(std::shared_ptr<Op> node) {
  LOG(FATAL) << "Unimplemented";
}

void Graph::cluster_gemm(std::shared_ptr<Op> node) {
  node->set_cluster_type(ClusterType::Gemm);
  node->set_cluster_id(this->gemm_cluster_id);
}

void Graph::cluster_conv(std::shared_ptr<Op> node) {
  node->set_cluster_type(ClusterType::Conv);
  node->set_cluster_id(this->conv_cluster_id);
}

bool Graph::remain_acyclic_after_add_node_to_cluster(
    std::shared_ptr<Op> node_to_add, ClusterType cluster_type, int cluster_id) {
  std::function<bool(std::shared_ptr<Op>, std::unordered_set<std::string>&)>
      dfs = [&](std::shared_ptr<Op> node,
                std::unordered_set<std::string>& visit) -> bool {
    if (visit.find(node->get_name()) != visit.end()) return true;
    visit.insert(node->get_name());

    for (auto const& edge : this->edges[node->get_name()]) {
      if (edge->get_dst()->get_name() == node_to_add->get_name() ||
          (edge->get_dst()->get_cluster_type() == cluster_type &&
           edge->get_dst()->get_cluster_id() == cluster_id))
        return false;

      if (!dfs(edge->get_dst(), visit)) return false;
    }

    return true;
  };

  std::unordered_set<std::string> visit;

  for (auto const& [node_name, node] : this->nodes) {
    if (node->get_cluster_type() != cluster_type) continue;
    if (node->get_cluster_id() != cluster_id) continue;

    for (auto const& edge : this->edges[node_name]) {
      if (edge->get_dst()->get_name() == node_to_add->get_name()) continue;
      if (edge->get_dst()->get_cluster_type() == cluster_type &&
          edge->get_dst()->get_cluster_id() == cluster_id)
        continue;

      visit.clear();
      if (!dfs(edge->get_dst(), visit)) return false;
    }
  }

  for (auto const& edge : this->edges[node_to_add->get_name()]) {
    if (edge->get_dst()->get_name() == node_to_add->get_name()) continue;
    if (edge->get_dst()->get_cluster_type() == cluster_type &&
        edge->get_dst()->get_cluster_id() == cluster_id)
      continue;

    visit.clear();
    if (!dfs(edge->get_dst(), visit)) return false;
  }

  return true;
}
}  // namespace graph
}  // namespace core
}  // namespace mononn_engine