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

#include "mononn_engine/optimization/clustering_single_node_pass.h"

#include "mononn_engine/config/config.h"
#include "mononn_engine/core/op/cluster_elewise.h"
#include "mononn_engine/core/op/cluster_reduce.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using Graph = mononn_engine::core::graph::Graph;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using Parameter = mononn_engine::core::op::Parameter;
using Constant = mononn_engine::core::op::Constant;
using Config = mononn_engine::config::Config;

std::string ClusteringSingleNodePass::name() const {
  return PassName::ClusteringSingleNodePass;
}

bool ClusteringSingleNodePass::run(Graph* graph,
                                   std::shared_ptr<CUDAContext> cuda_context) {
  for (auto node_name : graph->get_node_list()) {
    // scalar constant may be already deleted
    if (!graph->has_node(node_name)) {
      LOG(WARNING) << "ClusteringSingleNodePass::run not found node "
                   << node_name;
      continue;
    }

    auto node = graph->get_node(node_name);

    if (node->get_type() == OpType::custom_call ||
        node->get_type() == OpType::cluster ||
        node->get_type() == OpType::get_tuple_element ||
        node->get_type() == OpType::constant ||
        node->get_type() == OpType::parameter) {
      continue;
    }

    std::string new_cluster_name = "f_" + node_name;
    // std::string new_cluster_name = node_name;
    std::vector<std::shared_ptr<Op>> cluster_operands;

    for (int idx = 0; idx < node->get_operand_count(); ++idx) {
      if (node->get_operand(idx)->get_type() == OpType::constant &&
          node->get_operand(idx)->as<Constant>()->is_scalar()) {
        continue;
      }

      cluster_operands.push_back(node->get_operand(idx));
    }

    std::vector<TensorSpec> output_spec_list = node->get_output_specs();

    std::shared_ptr<ClusterOp> cluster_node;

    if (node->get_type() == OpType::reduce) {
      cluster_node = std::make_shared<ClusterReduce>(
          new_cluster_name, cluster_operands, output_spec_list);
    } else {
      cluster_node = std::make_shared<ClusterElewise>(
          new_cluster_name, cluster_operands, output_spec_list);
    }

    graph->add_node(cluster_node);

    cluster_node->set_hlo_text(node->get_hlo_text());

    std::shared_ptr<Graph> new_graph = std::make_shared<Graph>();
    node->set_attribute(OpAttribute::initial_cluster_tag,
                        cluster_node->get_name());
    node->set_attribute(OpAttribute::sub_cluster_tag, cluster_node->get_name());
    node->set_attribute(OpAttribute::sub_cluster_type,
                        cluster_node->get_cluster_type().to_string());
    node->set_attribute(OpAttribute::initial_cluster_type,
                        cluster_node->get_cluster_type().to_string());

    new_graph->add_node(node);

    for (int idx = 0; idx < (int)cluster_operands.size(); ++idx) {
      std::string parameter_node_name =
          "param_" + std::to_string(idx) + "_" + new_cluster_name;
      std::vector<TensorSpec> operand_output_spec =
          cluster_operands[idx]->get_output_specs();
      std::shared_ptr<Parameter> parameter_node = std::make_shared<Parameter>(
          parameter_node_name, std::vector<std::shared_ptr<Op>>{},
          operand_output_spec);
      parameter_node->set_parameter_number(idx);

      parameter_node->set_attribute(OpAttribute::initial_cluster_tag,
                                    cluster_node->get_name());
      parameter_node->set_attribute(OpAttribute::sub_cluster_tag,
                                    cluster_node->get_name());
      parameter_node->set_attribute(
          OpAttribute::sub_cluster_type,
          cluster_node->get_cluster_type().to_string());
      parameter_node->set_attribute(
          OpAttribute::initial_cluster_type,
          cluster_node->get_cluster_type().to_string());

      new_graph->add_node(parameter_node);
      new_graph->add_edge(parameter_node_name, node_name);
      new_graph->mark_as_input_node(parameter_node_name);

      node->replace_operand(cluster_operands[idx]->get_name(), parameter_node);
    }

    for (int idx = 0; idx < node->get_operand_count(); ++idx) {
      if (node->get_operand(idx)->get_type() == OpType::constant &&
          node->get_operand(idx)->as<Constant>()->is_scalar()) {
        auto node_constant_scalar = node->get_operand(idx);
        std::string new_constant_node_name =
            node_constant_scalar->get_name() + "_" + node_name;
        std::shared_ptr<Constant> new_constant_node =
            std::make_shared<Constant>(
                new_constant_node_name, std::vector<std::shared_ptr<Op>>{},
                node_constant_scalar->get_output_specs());

        new_constant_node->set_value(node_constant_scalar->get_value());
        new_constant_node->set_hlo_text(node_constant_scalar->get_hlo_text());

        new_constant_node->set_attribute(OpAttribute::initial_cluster_tag,
                                         cluster_node->get_name());
        new_constant_node->set_attribute(OpAttribute::sub_cluster_tag,
                                         cluster_node->get_name());
        new_constant_node->set_attribute(
            OpAttribute::sub_cluster_type,
            cluster_node->get_cluster_type().to_string());
        new_constant_node->set_attribute(
            OpAttribute::initial_cluster_type,
            cluster_node->get_cluster_type().to_string());

        new_graph->add_node(new_constant_node);
        new_graph->mark_as_extended_input_node(new_constant_node_name);
        new_graph->add_edge(new_constant_node, node);
        node->replace_operand(node_constant_scalar->get_name(),
                              new_constant_node);
      }
    }

    // replace edges
    for (auto const& edge : graph->get_node_input_edges(node_name)) {
      if (edge->get_src()->get_type() == OpType::constant &&
          edge->get_src()->as<Constant>()->is_scalar()) {
        graph->remove_edge(edge->get_src_name(), edge->get_dst_name());
        graph->add_control_edge(edge->get_src_name(), new_cluster_name);

        // We need to to keep scalar in the entry computation constant to build
        // hlo instruction schedule This dead node will participate node codegen
        // phase, but it seems ok. if
        // (graph->get_node_output_edges(edge->get_src_name()).size() == 0) {
        //     graph->remove_node(edge->get_src_name());
        // }

        continue;
      }

      graph->add_edge(edge->get_src_name(), new_cluster_name);
      graph->remove_edge(edge->get_src_name(), edge->get_dst_name());
    }

    for (auto const& edge : graph->get_node_output_edges(node_name)) {
      graph->add_edge(new_cluster_name, edge->get_dst_name());
      edge->get_dst()->replace_operand(node_name, cluster_node);
      graph->remove_edge(edge->get_src(), edge->get_dst());
    }

    // mark input output nodes
    for (auto const& node_name_new_graph : new_graph->get_node_list()) {
      auto node_new_graph = new_graph->get_node(node_name_new_graph);

      if (node_new_graph->get_type() == OpType::iota) {
        new_graph->mark_as_extended_input_node(node_name_new_graph);
      }
    }

    new_graph->mark_as_output_node(node_name);

    if (node->get_type() == OpType::gather) {
      new_graph->add_control_edge(node->get_operand(1), node->get_operand(0));
    }

    if (graph->is_output_node(node_name)) {
      int output_node_idx = graph->find_output_node_idx(node_name);
      graph->update_output_node(output_node_idx, cluster_node->get_name());
    }

    graph->remove_node(node_name);

    cluster_node->set_graph(new_graph);
    cluster_node->add_hlo_instruction_name(node->get_hlo_instruction_name());

    LOG(INFO) << "Clustering node " << node_name << " into cluster "
              << new_cluster_name;
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine