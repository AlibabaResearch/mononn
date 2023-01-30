#include "mononn_engine/core/graph/cluster_util.h"

#include <memory>

#include "mononn_engine/config/config.h"
#include "mononn_engine/core/edge/control_edge.h"
#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/op/get_tuple_element.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/tensor/tensor_spec.h"
#include "mononn_engine/helpers/macros.h"

namespace mononn_engine {
namespace core {
namespace graph {
using Op = mononn_engine::core::op::Op;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using Parameter = mononn_engine::core::op::Parameter;
using GetTupleElement = mononn_engine::core::op::GetTupleElement;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using Edge = mononn_engine::core::edge::Edge<Op>;
using ControlEdge = mononn_engine::core::edge::ControlEdge;
using Config = mononn_engine::config::Config;

template <typename T>
void maintain_hlo_instruction_name_list_merge_independent(
    std::shared_ptr<T> new_cluster, std::shared_ptr<ClusterOp> cluster1,
    std::shared_ptr<ClusterOp> cluster2) {
  for (auto const& inst_name : cluster1->get_hlo_instruction_name_list()) {
    new_cluster->add_hlo_instruction_name(inst_name);
  }

  for (auto const& inst_name : cluster2->get_hlo_instruction_name_list()) {
    new_cluster->add_hlo_instruction_name(inst_name);
  }
}

template <typename T>
void maintain_hlo_instruction_name_list_merge_sequential(
    std::shared_ptr<T> new_cluster, std::shared_ptr<ClusterOp> begin_cluster,
    std::vector<std::shared_ptr<Op>> nodes_in_path,
    std::shared_ptr<ClusterOp> end_cluster) {
  for (auto const& inst_name : begin_cluster->get_hlo_instruction_name_list()) {
    new_cluster->add_hlo_instruction_name(inst_name);
  }

  // for (auto const &node : nodes_in_path) {
  //     new_cluster->add_hlo_instruction_name(node->get_hlo_instruction_name());
  // }

  for (auto const& inst_name : end_cluster->get_hlo_instruction_name_list()) {
    new_cluster->add_hlo_instruction_name(inst_name);
  }
}

template <typename T>
std::shared_ptr<T> ClusterUtil::merge_independent(
    std::string cluster_name, std::shared_ptr<ClusterOp> cluster1,
    std::shared_ptr<ClusterOp> cluster2) {
  std::shared_ptr<T> new_cluster_node;
  std::vector<std::shared_ptr<Op>> new_operand_list;
  std::shared_ptr<Graph> new_graph = std::make_shared<Graph>();

  // add operands
  for (auto const& operand : cluster1->get_operands()) {
    new_operand_list.push_back(operand);
  }

  for (auto const& operand : cluster2->get_operands()) {
    new_operand_list.push_back(operand);
  }

  std::vector<TensorSpec> new_output_spec_list;

  // add outputs
  for (auto const& spec : cluster1->get_output_specs()) {
    new_output_spec_list.push_back(spec);
  }

  for (auto const& spec : cluster2->get_output_specs()) {
    new_output_spec_list.push_back(spec);
  }

  new_cluster_node =
      std::make_shared<T>(cluster_name, new_operand_list, new_output_spec_list);
  new_cluster_node->set_hlo_text(cluster1->get_hlo_text() + "\n//" +
                                 cluster2->get_hlo_text());

  // add nodes
  for (auto const& node_name : cluster1->get_graph()->get_node_list()) {
    std::shared_ptr<Op> node = cluster1->get_graph()->get_node(node_name);
    node->set_attribute(OpAttribute::sub_cluster_tag, cluster_name);
    node->set_attribute(OpAttribute::sub_cluster_type,
                        ClusterTypeOf<T>::Type.to_string());
    new_graph->add_node(node);
  }

  for (auto const& node_name : cluster2->get_graph()->get_node_list()) {
    std::shared_ptr<Op> node = cluster2->get_graph()->get_node(node_name);
    node->set_attribute(OpAttribute::sub_cluster_tag, cluster_name);
    node->set_attribute(OpAttribute::sub_cluster_type,
                        ClusterTypeOf<T>::Type.to_string());
    new_graph->add_node(node);
  }

  // add edges
  for (auto const& node_name : cluster1->get_graph()->get_node_list()) {
    for (auto const& edge :
         cluster1->get_graph()->get_node_output_edges(node_name)) {
      new_graph->add_edge(edge);
    }
  }

  for (auto const& node_name : cluster2->get_graph()->get_node_list()) {
    for (auto const& edge :
         cluster2->get_graph()->get_node_output_edges(node_name)) {
      new_graph->add_edge(edge);
    }
  }

  // add control edges
  for (auto const& node_name : cluster1->get_graph()->get_node_list()) {
    for (auto const& edge :
         cluster1->get_graph()->get_node_output_control_edges(node_name)) {
      new_graph->add_control_edge(edge);
    }
  }

  for (auto const& node_name : cluster2->get_graph()->get_node_list()) {
    for (auto const& edge :
         cluster2->get_graph()->get_node_output_control_edges(node_name)) {
      new_graph->add_control_edge(edge);
    }
  }

  // mark input nodes
  for (auto const& node_name : cluster1->get_graph()->get_input_nodes()) {
    new_graph->mark_as_input_node(node_name);
  }

  for (auto const& node_name : cluster2->get_graph()->get_input_nodes()) {
    new_graph->mark_as_input_node(node_name);
  }

  new_graph->align_input_nodes();

  // mark extended input nodes
  for (auto const& node_name :
       cluster1->get_graph()->get_extended_input_nodes()) {
    new_graph->mark_as_extended_input_node(node_name);
  }

  for (auto const& node_name :
       cluster2->get_graph()->get_extended_input_nodes()) {
    new_graph->mark_as_extended_input_node(node_name);
  }

  // mark output nodes
  for (auto const& node_name : cluster1->get_graph()->get_output_nodes()) {
    new_graph->mark_as_output_node(node_name);
  }

  for (auto const& node_name : cluster2->get_graph()->get_output_nodes()) {
    new_graph->mark_as_output_node(node_name);
  }

  // Add control dependency to reduce node;
  if (cluster1->is_cluster_reduce()) {
    std::vector<Op*> reduce_node_list =
        cluster1->as<ClusterReduce>()->get_reduce_nodes();

    for (auto const& node_name : cluster2->get_graph()->get_node_list()) {
      if (cluster2->get_graph()->get_node(node_name)->get_type() ==
          OpType::reduce)
        continue;

      for (auto const& reduce_node : reduce_node_list) {
        new_graph->add_control_edge(node_name, reduce_node->get_name());
      }
    }
  }

  if (cluster2->is_cluster_reduce()) {
    std::vector<Op*> reduce_node_list =
        cluster2->as<ClusterReduce>()->get_reduce_nodes();

    for (auto const& node_name : cluster1->get_graph()->get_node_list()) {
      if (cluster1->get_graph()->get_node(node_name)->get_type() ==
          OpType::reduce)
        continue;

      for (auto const& reduce_node : reduce_node_list) {
        new_graph->add_control_edge(node_name, reduce_node->get_name());
      }
    }
  }

  new_cluster_node->set_graph(new_graph);
  maintain_hlo_instruction_name_list_merge_independent<T>(new_cluster_node,
                                                          cluster1, cluster2);

  if (Config::get()->run_expensive_verification) {
    std::vector<std::string> node_list;
    if (!new_graph->is_acyclic(node_list, true)) {
      LOG(FATAL) << "Detected cycle in graph: "
                 << mononn_engine::helpers::join(" ", node_list);
    }
  }

  return new_cluster_node;
}

template <typename T>
std::shared_ptr<T> ClusterUtil::merge_sequential(
    std::string cluster_name, std::shared_ptr<ClusterOp> begin_cluster,
    std::vector<std::shared_ptr<Op>> nodes_in_path,
    std::shared_ptr<ClusterOp> end_cluster, Graph* graph) {
  for (auto const& node : nodes_in_path) {
    if (node->get_type() != OpType::get_tuple_element) {
      LOG(FATAL) << "Only get tuple node is supported in path. Get "
                 << node->get_type().to_string();
    }
  }

  std::shared_ptr<T> new_cluster_node;
  std::shared_ptr<Graph> new_graph = std::make_shared<Graph>();
  std::vector<std::shared_ptr<Op>> new_cluster_operand_list;
  std::vector<TensorSpec> new_cluster_output_spec_list;
  std::vector<std::string> new_cluster_output_node_list;
  std::vector<std::string> new_cluster_input_node_list;
  begin_cluster->get_graph()->build_transitive_closure();

  //        std::vector<std::string> begin_cluster_graph_input_nodes =
  //        begin_cluster->get_graph()->get_input_nodes();
  //        std::vector<std::string> begin_cluster_graph_output_nodes =
  //        begin_cluster->get_graph()->get_output_nodes();
  //        std::vector<std::string> end_cluster_graph_input_nodes =
  //        end_cluster->get_graph()->get_input_nodes();
  //        std::vector<std::string> end_cluster_graph_output_nodes =
  //        end_cluster->get_graph()->get_output_nodes();
  //
  // add nodes + edges for begin cluster
  for (auto const& node_name : begin_cluster->get_graph()->get_node_list()) {
    new_graph->add_node(begin_cluster->get_graph()->get_node(node_name));
  }

  for (auto const& node_name : begin_cluster->get_graph()->get_node_list()) {
    for (auto const& edge :
         begin_cluster->get_graph()->get_node_output_edges(node_name)) {
      new_graph->add_edge(edge);
    }

    for (auto const& edge :
         begin_cluster->get_graph()->get_node_output_control_edges(node_name)) {
      new_graph->add_control_edge(edge);
    }
  }

  // add nodes + edges for end cluster
  for (auto const& node_name : end_cluster->get_graph()->get_node_list()) {
    new_graph->add_node(end_cluster->get_graph()->get_node(node_name));
  }

  for (auto const& node_name : end_cluster->get_graph()->get_node_list()) {
    for (auto const& edge :
         end_cluster->get_graph()->get_node_output_edges(node_name)) {
      new_graph->add_edge(edge);
    }

    for (auto const& edge :
         end_cluster->get_graph()->get_node_output_control_edges(node_name)) {
      new_graph->add_control_edge(edge);
    }
  }

  // mark input nodes
  EXPECT_TRUE(begin_cluster->get_operand_count() ==
                  begin_cluster->get_graph()->get_input_node_count(),
              "count mismatch for " + begin_cluster->get_name());
  EXPECT_TRUE(end_cluster->get_operand_count() ==
                  end_cluster->get_graph()->get_input_node_count(),
              "count mismatch for " + end_cluster->get_name());
  EXPECT_TRUE(begin_cluster->get_output_specs_count() ==
                  begin_cluster->get_graph()->get_output_node_count(),
              "count mismatch");
  EXPECT_TRUE(end_cluster->get_output_specs_count() ==
                  end_cluster->get_graph()->get_output_node_count(),
              "count mismatch");

  for (auto const& node_name : begin_cluster->get_graph()->get_input_nodes()) {
    new_graph->mark_as_input_node(node_name);
  }

  for (auto const& node_name :
       begin_cluster->get_graph()->get_extended_input_nodes()) {
    new_graph->mark_as_extended_input_node(node_name);
  }

  // assume operand list match with input node list
  for (int idx = 0; idx < end_cluster->get_graph()->get_input_node_count();
       ++idx) {
    std::shared_ptr<Op> operand = end_cluster->get_operand(idx);
    std::string input_node_name = end_cluster->get_graph()->get_input_node(idx);
    std::shared_ptr<Op> input_node =
        end_cluster->get_graph()->get_node(input_node_name);

    if (operand->get_name() == begin_cluster->get_name()) {
      input_node->set_attribute(OpAttribute::on_chip_transfer_from_node,
                                begin_cluster->get_graph()->get_output_node(0));
      continue;
    } else if (operand->get_type() == OpType::get_tuple_element &&
               operand->get_operand(0)->get_name() ==
                   begin_cluster->get_name()) {
      input_node->set_attribute(
          OpAttribute::on_chip_transfer_from_node,
          begin_cluster->get_graph()->get_output_node(
              operand->as<GetTupleElement>()->get_tuple_index()));
      continue;
    }

    new_graph->mark_as_input_node(input_node_name);
  }

  for (auto const& node_name :
       end_cluster->get_graph()->get_extended_input_nodes()) {
    new_graph->mark_as_extended_input_node(node_name);
  }

  new_graph->align_input_nodes();

  // add control dependency for data dependency
  if (nodes_in_path.empty()) {  // two cluster directly connected
    EXPECT_TRUE(begin_cluster->get_graph()->get_output_node_count() == 1,
                "Begin cluster " + begin_cluster->get_name() + "have " +
                    std::to_string(
                        begin_cluster->get_graph()->get_output_node_count()) +
                    " output node(s).");

    int idx = 0;
    for (; idx < end_cluster->get_operand_count(); ++idx) {
      if (end_cluster->get_operand(idx)->get_name() ==
          begin_cluster->get_name())
        break;
    }

    if (idx == end_cluster->get_operand_count())
      LOG(FATAL) << "Not found node " << begin_cluster->get_name() << " in "
                 << end_cluster->get_name() << "'s operands";

    // add operand and output
    for (auto const& operand : begin_cluster->get_operands()) {
      new_cluster_operand_list.push_back(operand);
    }

    for (auto const& operand : end_cluster->get_operands()) {
      if (operand->get_name() == end_cluster->get_operand(idx)->get_name())
        continue;
      new_cluster_operand_list.push_back(operand);
    }

    std::vector<std::string> begin_cluster_output_node_list =
        begin_cluster->get_graph()->get_output_nodes();
    //            std::vector<Op*> begin_cluster_reduce_node_list =
    //            begin_cluster->as<ClusterReduce>()->get_reduce_nodes_in_last_sub_cluster();
    EXPECT_TRUE(begin_cluster_output_node_list.size() == 1,
                "Begin cluster should have only one output node because they "
                "are directly connected (not via get tuple element node), got" +
                    std::to_string(begin_cluster_output_node_list.size()));
    //            Op *begin_cluster_output_node =
    //            begin_cluster_reduce_node_list[0];

    // Add to new cluster output node if begin cluster's output is *not only*
    // used by end cluster.
    if (graph->get_node_output_edges(begin_cluster->get_name()).size() > 1) {
      for (auto const& begin_cluster_output_node :
           begin_cluster_output_node_list) {
        new_cluster_output_spec_list.push_back(
            begin_cluster->get_graph()
                ->get_node(begin_cluster_output_node)
                ->get_output_spec(0));
        new_cluster_output_node_list.push_back(begin_cluster_output_node);
      }
    }

    for (auto const& node_name : end_cluster->get_graph()->get_output_nodes()) {
      new_cluster_output_spec_list.push_back(
          end_cluster->get_graph()->get_node(node_name)->get_output_spec(0));
      new_cluster_output_node_list.push_back(node_name);
    }
  } else {  // two clusters connected via get tuple element node
    if (begin_cluster->get_output_specs_count() < 2) {
      LOG(FATAL) << "Begin cluster " << begin_cluster->get_name()
                 << " have less than two outputs.";
    }

    // add operand and output
    for (auto const& operand : begin_cluster->get_operands()) {
      new_cluster_operand_list.push_back(operand);
    }

    for (auto const& operand : end_cluster->get_operands()) {
      if (std::find_if(nodes_in_path.begin(), nodes_in_path.end(),
                       [&](std::shared_ptr<Op> op) -> bool {
                         return operand->get_name() == op->get_name();
                       }) != nodes_in_path.end())
        continue;
      new_cluster_operand_list.push_back(operand);
    }

    std::vector<std::string> begin_cluster_output_nodes =
        begin_cluster->get_graph()->get_output_nodes();

    // mark begin cluster's output node as output node in new graph if and only
    // if
    // 1. output node consumed by cluster other than end cluster. or
    // 2. output node consumed by not only end cluster but also other
    // cluster(s).
    for (int idx = 0; idx < (int)begin_cluster_output_nodes.size(); ++idx) {
      std::vector<std::shared_ptr<Op>>::iterator iter = std::find_if(
          nodes_in_path.begin(), nodes_in_path.end(),
          [&](std::shared_ptr<Op> op) -> bool {
            return idx == op->as<GetTupleElement>()->get_tuple_index();
          });

      bool is_output = false;
      if (iter == nodes_in_path.end()) {
        is_output = true;
      } else {
        if (graph->get_node_output_edges((*iter)->get_name()).size() > 1) {
          is_output = true;
        }
      }

      if (is_output) {
        new_cluster_output_spec_list.push_back(
            begin_cluster->get_graph()
                ->get_node(begin_cluster_output_nodes[idx])
                ->get_output_spec(0));
        new_cluster_output_node_list.push_back(begin_cluster_output_nodes[idx]);
      }
    }

    for (auto const& node_name : end_cluster->get_graph()->get_output_nodes()) {
      new_cluster_output_spec_list.push_back(
          end_cluster->get_graph()->get_node(node_name)->get_output_spec(0));
      new_cluster_output_node_list.push_back(node_name);
    }
  }

  for (auto const& node_name : new_cluster_output_node_list) {
    new_graph->mark_as_output_node(node_name);
  }

  std::vector<std::string> begin_cluster_output_nodes =
      begin_cluster->get_graph()->get_output_nodes();

  // add control dependency
  for (auto const& output_node_name : begin_cluster_output_nodes) {
    // add control edge
    for (auto const& node_name : end_cluster->get_graph()->get_input_nodes()) {
      new_graph->add_control_edge(output_node_name, node_name);
    }

    for (auto const& node_name :
         end_cluster->get_graph()->get_extended_input_nodes()) {
      new_graph->add_control_edge(output_node_name, node_name);
    }
  }

  new_cluster_node = std::make_shared<T>(cluster_name, new_cluster_operand_list,
                                         new_cluster_output_spec_list);

  new_cluster_node->set_graph(new_graph);

  {
    // Maintain sub cluster order
    // This should happen if:
    // 1. New cluster is elewise cluster: no-need maintain
    // 2. New cluster is reduce cluster, first cluster is elewise cluster or its
    // has a elewise tailing sub-cluster (i.e., maintained sub-cluster
    // tag/type),
    //      new order is concatenation of two clusters' order. However
    //      intermediate sub-clusters should be merged.
    // 3. New cluster is reduce cluster, first cluster is reduce cluster with a
    // reduce tailing sub-cluster.
    //      New order is concatenation of two clusters' order.

    new_cluster_node->set_sub_cluster_tag_order({});
    new_cluster_node->set_sub_cluster_type_order({});

    if (begin_cluster->get_sub_cluster_type_order().back() ==
        ClusterType::Elewise.to_string()) {
      int sub_cluster_count1 =
          begin_cluster->get_sub_cluster_tag_order().size();
      int sub_cluster_count2 = end_cluster->get_sub_cluster_tag_order().size();

      std::string first_cluster_tail_sub_cluster_tag =
          begin_cluster->get_sub_cluster_tag_order().back();
      std::string end_cluster_front_sub_cluster_tag =
          end_cluster->get_sub_cluster_tag_order().front();
      std::string new_sub_cluster_tag = first_cluster_tail_sub_cluster_tag +
                                        "_TagMD_" +
                                        end_cluster_front_sub_cluster_tag;

      if (new_cluster_node->is_cluster_elewise()) {
        new_sub_cluster_tag = cluster_name;
      }

      std::string new_sub_cluster_type;
      if (end_cluster->get_sub_cluster_type_order().front() ==
          ClusterType::Elewise.to_string()) {
        new_sub_cluster_type = ClusterType::Elewise.to_string();
      } else {
        new_sub_cluster_type = ClusterType::Reduce.to_string();
      }

      for (int idx = 0; idx < sub_cluster_count1; ++idx) {
        std::string sub_cluster_tag =
            begin_cluster->get_sub_cluster_tag_order()[idx];
        std::string sub_cluster_type =
            begin_cluster->get_sub_cluster_type_order()[idx];
        if (sub_cluster_tag != first_cluster_tail_sub_cluster_tag) {
          new_cluster_node->append_sub_cluster_tag(sub_cluster_tag);
          new_cluster_node->append_sub_cluster_type(sub_cluster_type);
        }
      }

      new_cluster_node->append_sub_cluster_tag(new_sub_cluster_tag);
      new_cluster_node->append_sub_cluster_type(new_sub_cluster_type);

      for (int idx = 0; idx < sub_cluster_count2; ++idx) {
        std::string sub_cluster_tag =
            end_cluster->get_sub_cluster_tag_order()[idx];
        std::string sub_cluster_type =
            end_cluster->get_sub_cluster_type_order()[idx];

        if (sub_cluster_tag != end_cluster_front_sub_cluster_tag) {
          new_cluster_node->append_sub_cluster_tag(sub_cluster_tag);
          new_cluster_node->append_sub_cluster_type(sub_cluster_type);
        }
      }

      // Maintain sub cluster tag and sub cluster type for nodes
      // This should happen in two scenarios.
      // 1. First cluster in concatenation is elewise cluster
      // 2. First cluster in concatenation is reduce cluster with a tailing
      // elewise sub-cluser
      std::vector<std::string> parameter_nodes_to_remove;
      for (auto const& node_name : new_graph->get_node_list()) {
        auto node = new_graph->get_node(node_name);
        if (node->get_attribute(OpAttribute::sub_cluster_tag) ==
                first_cluster_tail_sub_cluster_tag ||
            node->get_attribute(OpAttribute::sub_cluster_tag) ==
                end_cluster_front_sub_cluster_tag) {
          if (node->get_attribute(OpAttribute::sub_cluster_tag) ==
                  end_cluster_front_sub_cluster_tag &&
              node->get_type() == OpType::parameter &&
              node->has_attribute(OpAttribute::on_chip_transfer_from_node)) {
            parameter_nodes_to_remove.push_back(node_name);

            EXPECT_TRUE(new_graph->get_node_output_edges(node_name).size() == 1,
                        "Parameter node should have only one output edge");

            std::string upstream_node_name =
                node->get_attribute(OpAttribute::on_chip_transfer_from_node);
            std::string downstream_node_name =
                new_graph->get_node_output_edges(node_name)[0]->get_dst_name();

            auto upstream_node = new_graph->get_node(upstream_node_name);
            auto downstream_node = new_graph->get_node(downstream_node_name);

            downstream_node->replace_operand(node_name, upstream_node);
            new_graph->add_edge(upstream_node, downstream_node);
            new_graph->remove_edge(node_name, downstream_node_name);

            std::vector<std::shared_ptr<ControlEdge>> input_control_edges =
                new_graph->get_node_input_control_edges(node_name);
            std::vector<std::shared_ptr<ControlEdge>> output_control_edges =
                new_graph->get_node_output_control_edges(node_name);

            {  // move control edge to downstream_node
              for (auto const& edge : input_control_edges) {
                new_graph->add_control_edge(edge->get_src_name(),
                                            downstream_node_name);
              }

              for (auto const& edge : output_control_edges) {
                new_graph->add_control_edge(downstream_node_name,
                                            edge->get_dst_name());
              }
            }

            {  // remove old control edge on node_name
              for (auto const& edge : input_control_edges) {
                new_graph->remove_control_edge(edge->get_src(),
                                               edge->get_dst());
              }

              for (auto const& edge : output_control_edges) {
                new_graph->remove_control_edge(edge->get_src(),
                                               edge->get_dst());
              }
            }

            continue;
          }

          node->set_attribute(OpAttribute::sub_cluster_tag,
                              new_sub_cluster_tag);
          node->set_attribute(OpAttribute::sub_cluster_type,
                              new_sub_cluster_type);
        }
      }

      for (auto const& node_name : parameter_nodes_to_remove) {
        new_graph->remove_node(node_name);
      }

    } else {
      new_cluster_node->append_sub_cluster_tags(
          begin_cluster->get_sub_cluster_tag_order());
      new_cluster_node->append_sub_cluster_tags(
          end_cluster->get_sub_cluster_tag_order());

      new_cluster_node->append_sub_cluster_types(
          begin_cluster->get_sub_cluster_type_order());
      new_cluster_node->append_sub_cluster_types(
          end_cluster->get_sub_cluster_type_order());
    }
  }

  maintain_hlo_instruction_name_list_merge_sequential<T>(
      new_cluster_node, begin_cluster, nodes_in_path, end_cluster);

  if (Config::get()->run_expensive_verification) {
    std::vector<std::string> node_list;
    if (!new_graph->is_acyclic(node_list, true)) {
      LOG(FATAL) << "Detected cycle in graph: "
                 << mononn_engine::helpers::join(" ", node_list);
    }
  }

  return new_cluster_node;
}

void ClusterUtil::replace_node(Graph* graph, std::vector<std::string> node_list,
                               std::shared_ptr<Op> new_node) {
  graph->add_node(new_node);

  if (std::any_of(node_list.begin(), node_list.end(),
                  [&](std::string const& node_name) -> bool {
                    return graph->is_output_node(node_name);
                  })) {
    graph->mark_as_output_node(new_node->get_name());
  }

  for (auto const& node_name : node_list) {
    std::vector<std::string> input_nodes = graph->get_input_nodes();
    std::vector<std::string>::iterator iter =
        std::find(input_nodes.begin(), input_nodes.end(), node_name);
    if (iter != input_nodes.end()) {
      LOG(FATAL) << "Cannot replace input nodes";
    }

    std::vector<std::string> extended_input_nodes =
        graph->get_extended_input_nodes();
    iter = std::find(extended_input_nodes.begin(), extended_input_nodes.end(),
                     node_name);
    if (iter != extended_input_nodes.end()) {
      LOG(FATAL) << "Cannot replace extended input nodes";
    }

    std::shared_ptr<Op> node = graph->get_node(node_name);
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

  std::function<int(const ClusterOp*, const GetTupleElement*, const ClusterOp*)>
      get_GTE_index_in_new_cluster = [&](const ClusterOp* old_cluster,
                                         const GetTupleElement* node,
                                         const ClusterOp* new_cluster) -> int {
    std::string output_node_name =
        old_cluster->get_graph()->get_output_node(node->get_tuple_index());
    int new_index =
        new_cluster->get_graph()->find_output_node_idx(output_node_name);
    return new_index;
  };

  std::vector<std::string> get_tuple_element_node_to_be_removed;
  for (auto const& node_name : node_list) {
    if (graph->get_node(node_name)->get_type() == OpType::get_tuple_element)
      continue;

    std::vector<std::shared_ptr<Edge>> node_input_edges =
        graph->get_node_input_edges(node_name);
    std::vector<std::shared_ptr<Edge>> node_output_edges =
        graph->get_node_output_edges(node_name);
    std::vector<std::shared_ptr<ControlEdge>> node_input_control_edges =
        graph->get_node_input_control_edges(node_name);
    std::vector<std::shared_ptr<ControlEdge>> node_output_control_edges =
        graph->get_node_output_control_edges(node_name);

    for (auto const& edge : node_input_edges) {
      graph->remove_edge(edge->get_src(), edge->get_dst());

      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<Edge> new_edge =
          std::make_shared<Edge>(edge->get_src(), new_node);
      new_edge->set_sync(edge->get_sync());
      graph->add_edge(new_edge);
    }

    for (auto const& edge : node_input_control_edges) {
      graph->remove_control_edge(edge->get_src(), edge->get_dst());
      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<ControlEdge> new_control_edge =
          std::make_shared<ControlEdge>(edge->get_src(), new_node);
      graph->add_control_edge(new_control_edge);
    }

    for (auto const& edge : node_output_control_edges) {
      graph->remove_control_edge(edge->get_src(), edge->get_dst());
      if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name()))
        continue;
      std::shared_ptr<ControlEdge> new_control_edge =
          std::make_shared<ControlEdge>(new_node, edge->get_dst());
      graph->add_control_edge(new_control_edge);
    }

    if (graph->get_node(node_name)->get_output_specs().size() > 1) {
      for (auto const& edge :
           node_output_edges) {  // for each get tuple element
        if (edge->get_dst()->get_type() != OpType::get_tuple_element) {
          LOG(FATAL) << "Node " << node_name
                     << " have multiple output spec but "
                     << edge->get_dst()->get_name()
                     << " is not get tuple element";
        }

        std::vector<std::shared_ptr<Edge>> old_get_tuple_element_output_edges =
            graph->get_node_output_edges(edge->get_dst()->get_name());

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
            graph->remove_edge(get_tuple_element_output_edge);
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

          int index_for_new_get_tuple_element = get_GTE_index_in_new_cluster(
              edge->get_src()->as<ClusterOp>(),
              edge->get_dst()->as<GetTupleElement>(),
              new_node->as<ClusterOp>());
          std::string new_get_tuple_element_name =
              "get_tuple_element_" + new_node_name + "_" +
              std::to_string(index_for_new_get_tuple_element);
          std::shared_ptr<GetTupleElement> get_tuple_element =
              std::make_shared<GetTupleElement>(
                  new_get_tuple_element_name,
                  std::vector<std::shared_ptr<Op>>{new_node},
                  std::vector<TensorSpec>{edge->get_dst()->get_output_spec(0)});
          get_tuple_element->set_tuple_index(index_for_new_get_tuple_element);
          graph->add_node(get_tuple_element);
          graph->add_edge(new_node, get_tuple_element);

          for (auto const& old_get_tuple_element_output_edge :
               old_get_tuple_element_output_edges) {
            if (edge_in_node_list(
                    old_get_tuple_element_output_edge->get_src_name(),
                    old_get_tuple_element_output_edge->get_dst_name())) {
              graph->remove_edge(old_get_tuple_element_output_edge);
              continue;
            }

            graph->add_edge(get_tuple_element,
                            old_get_tuple_element_output_edge->get_dst());
            old_get_tuple_element_output_edge->get_dst()->replace_operand(
                old_get_tuple_element_output_edge->get_src_name(),
                get_tuple_element);
            graph->remove_edge(old_get_tuple_element_output_edge);
          }
        }

        graph->remove_edge(edge);
        // remote get tuple element node if it not in node_list, as new get
        // tuple element nodes will replace them
        if (std::find(node_list.begin(), node_list.end(),
                      edge->get_dst_name()) == node_list.end()) {
          graph->remove_node(edge->get_dst());
        }
      }
    } else {
      EXPECT_TRUE(
          graph->get_node(node_name)
                  ->as<ClusterOp>()
                  ->get_graph()
                  ->get_output_node_count() == 1,
          "Cluster " + node_name +
              " should have only one output in graph, got " +
              mononn_engine::helpers::join(", ", graph->get_node(node_name)
                                                     ->as<ClusterOp>()
                                                     ->get_graph_ptr()
                                                     ->get_output_nodes()));

      // all output edges are in the node list to be merged
      if (std::all_of(node_output_edges.begin(), node_output_edges.end(),
                      [&](std::shared_ptr<Edge> edge) -> bool {
                        return std::find(node_list.begin(), node_list.end(),
                                         edge->get_dst_name()) !=
                               node_list.end();
                      })) {
        for (auto const& edge : node_output_edges) {
          graph->remove_edge(edge);
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
          std::string output_node_name_for_old_cluster =
              graph->get_node(node_name)
                  ->as<ClusterOp>()
                  ->get_graph()
                  ->get_output_node(0);
          int index_for_new_get_tuple_element =
              new_node->as<ClusterOp>()->get_graph()->find_output_node_idx(
                  output_node_name_for_old_cluster);

          std::string new_get_tuple_element_name =
              "get_tuple_element_" + new_node_name + "_" +
              std::to_string(index_for_new_get_tuple_element);
          std::shared_ptr<GetTupleElement> get_tuple_element =
              std::make_shared<GetTupleElement>(
                  new_get_tuple_element_name,
                  std::vector<std::shared_ptr<Op>>{new_node},
                  std::vector<TensorSpec>{
                      graph->get_node(node_name)->get_output_spec(0)});
          get_tuple_element->set_tuple_index(index_for_new_get_tuple_element);
          graph->add_node(get_tuple_element);
          graph->add_edge(new_node, get_tuple_element);

          for (auto const& edge : node_output_edges) {
            if (edge_in_node_list(edge->get_src_name(), edge->get_dst_name())) {
              graph->remove_edge(edge);
              continue;
            }

            graph->add_edge(get_tuple_element, edge->get_dst());
            edge->get_dst()->replace_operand(node_name, get_tuple_element);
            graph->remove_edge(edge);
          }
        } else {
          for (auto const& edge : node_output_edges) {
            EXPECT_FALSE(
                edge_in_node_list(edge->get_src_name(), edge->get_dst_name()),
                "Edge " + edge->to_string() + " should not in node list: " +
                    mononn_engine::helpers::join(" ", node_list));

            graph->add_edge(new_node, edge->get_dst());
            edge->get_dst()->replace_operand(node_name, new_node);
            graph->remove_edge(edge);
          }
        }
      }
    }
  }

  // remove all merged nodes.
  for (auto const& node_name : node_list) {
    graph->remove_node(node_name);
  }

  if (Config::get()->run_expensive_verification) {
    std::vector<std::string> node_list;
    if (!graph->is_acyclic(node_list, true)) {
      LOG(FATAL) << "Detected cycle in graph: "
                 << mononn_engine::helpers::join(" ", node_list);
    }
  }
}

void ClusterUtil::summary_graph(Graph* graph) {
  LOG(INFO) << graph->summary();

  for (auto const& node_name : graph->get_node_list()) {
    auto node = graph->get_node(node_name);
    if (node->get_type() == OpType::cluster) {
      LOG(INFO) << "~~~~~~~~~~~~~~~~~~~~~~~~Summary for cluster " << node_name
                << "~~~~~~~~~~~~~~~~~~~~~~~~";
      LOG(INFO) << node->as<ClusterOp>()->get_graph_ptr()->summary();
    }
  }
}

std::unique_ptr<Graph> ClusterUtil::deep_copy_graph(const Graph* graph) {}

template std::shared_ptr<ClusterElewise>
ClusterUtil::merge_independent<ClusterElewise>(
    std::string cluster_name, std::shared_ptr<ClusterOp> cluster1,
    std::shared_ptr<ClusterOp> cluster2);
template std::shared_ptr<ClusterReduce>
ClusterUtil::merge_independent<ClusterReduce>(
    std::string cluster_name, std::shared_ptr<ClusterOp> cluster1,
    std::shared_ptr<ClusterOp> cluster2);

template std::shared_ptr<ClusterElewise>
ClusterUtil::merge_sequential<ClusterElewise>(
    std::string cluster_name, std::shared_ptr<ClusterOp> begin_cluster,
    std::vector<std::shared_ptr<Op>> nodes_in_path,
    std::shared_ptr<ClusterOp> end_cluster, Graph* graph);
template std::shared_ptr<ClusterReduce>
ClusterUtil::merge_sequential<ClusterReduce>(
    std::string cluster_name, std::shared_ptr<ClusterOp> begin_cluster,
    std::vector<std::shared_ptr<Op>> nodes_in_path,
    std::shared_ptr<ClusterOp> end_cluster, Graph* graph);

const ClusterType ClusterTypeOf<ClusterElewise>::Type("Elewise");
const ClusterType ClusterTypeOf<ClusterReduce>::Type("Reduce");
}  // namespace graph
}  // namespace core
}  // namespace mononn_engine