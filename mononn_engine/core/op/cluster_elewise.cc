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

#include "mononn_engine/core/op/cluster_elewise.h"

#include <sstream>
#include <unordered_map>

#include "mononn_engine/codegen/cuda_emitter.h"
#include "mononn_engine/core/context/index_tracer.h"
#include "mononn_engine/core/context/index_transform.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/gpu/smem_manager.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/op_impl/pad_impl.h"
#include "mononn_engine/core/schedule/loop.h"
#include "mononn_engine/core/schedule/schedule_factory.h"
#include "tensorflow/core/platform/logging.h"
namespace mononn_engine {
namespace core {
namespace op {
using Loop = mononn_engine::core::schedule::Loop;
using Schedule = mononn_engine::core::schedule::Schedule;
using ScheduleFactory = mononn_engine::core::schedule::ScheduleFactory;
using TensorShape = mononn_engine::core::tensor::TensorShape;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Memory = mononn_engine::core::gpu::Memory;
using IndexTracer = mononn_engine::core::context::IndexTracer;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using OpType = mononn_engine::core::op::OpType;
using PadImpl = mononn_engine::core::op_impl::PadImpl;
using IndexTransform = mononn_engine::core::context::IndexTransform;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using Graph = mononn_engine::core::graph::Graph;
using CUDAEmitter = mononn_engine::codegen::CUDAEmitter;

ClusterElewise::ClusterElewise(std::string _name,
                               std::vector<std::shared_ptr<Op>> _operands,
                               std::vector<TensorSpec> _output_specs)
    : ClusterOp(_name, _operands, _output_specs) {
  this->sub_cluster_type_order = {this->get_cluster_type().to_string()};
  this->sub_cluster_tag_order = {this->get_name()};
}

TensorShape ClusterElewise::get_loop_shape() const {
  LOG_ONCE(WARNING, __already_logged, "Need take tuple into account.");

  std::string output_node1_name = this->get_graph()->get_output_nodes()[0];
  TensorShape loop_shape = this->get_graph()
                               ->get_node(output_node1_name)
                               ->get_output_spec(0)
                               .get_shape();

  for (auto const& output_node_name : this->get_graph()->get_output_nodes()) {
    if (loop_shape.element_count() != this->get_graph()
                                          ->get_node(output_node_name)
                                          ->get_output_spec(0)
                                          .get_shape()
                                          .element_count()) {
      LOG(WARNING) << "Mismatched output shape in cluster " << this->get_name()
                   << " " << output_node1_name << " " << loop_shape.to_string()
                   << " " << output_node_name << " "
                   << this->get_graph()
                          ->get_node(output_node_name)
                          ->get_output_spec(0)
                          .get_shape()
                          .to_string();
    }
  }

  bool valid = true;
  for (auto const& node_name : this->get_graph()->get_output_nodes()) {
    if (loop_shape.element_count() < this->get_graph()
                                         ->get_node(node_name)
                                         ->get_output_spec(0)
                                         .get_shape()
                                         .element_count()) {
      loop_shape = this->get_graph()
                       ->get_node(node_name)
                       ->get_output_spec(0)
                       .get_shape();
    }
  }

  return loop_shape;
}

Schedule ClusterElewise::construct_schedule(LocalityTier::Tier tier) {
  EXPECT_TRUE(tier == LocalityTier::kT0,
              "Elewise cluster must have t0 locality tier");
  return ScheduleFactory::get_schedule(tier, this->get_loop_shape());
}

void ClusterElewise::trace_symbolic_index() {
  // trace symbolic index
  this->get_graph()->reverse_topology_order(
      [&](std::shared_ptr<const Op> node,
          std::shared_ptr<const Op> next_node) -> void {
        if (this->get_graph()->is_node_traced(node->get_name())) return;

        // this->get_graph()->trace_index(loop1_key, node->get_name());
        this->get_graph()->trace_index("{linear_index}", node->get_name());
      });
}

// bool should_use_new_code_emitter(const Graph *graph) {
//     for (auto const &node_name : graph->get_node_list()) {
//         auto node = graph->get_node(node_name);

//         if (node->get_type() == OpType::reduce_window) {
//             return true;
//         }
//     }

//     return false;
// }

// std::string emit_code_use_new_code_emitter(const Graph *graph, const
// std::string &loop_key, const std::string &loop_stride) {
//     using CUDAEmitter = mononn_engine::codegen::CUDAEmitter;
//     using CodegenStateSpaceMgr =
//     mononn_engine::codegen::CodegenStateSpaceMgr; using IndexSymbols =
//     mononn_engine::core::context::IndexSymbols;

//     CodegenStateSpaceMgr codege_state_mgr;
//     CUDAEmitter cuda_emitter(graph,
//     codege_state_mgr.get_codegen_state_space());
//     std::unordered_set<std::string> visit;

//     std::map<std::string, std::string> symbolic_index_initializer;

//     if (graph->get_instruction_parallel_factor() != 1) {
//         codege_state_mgr.emit_instruction_level_parallelism(graph->get_instruction_parallel_factor());
//         symbolic_index_initializer = {
//             {IndexSymbols::linear_index,
//             mononn_engine::helpers::string_format("((%s) + (%s *
//             {ilp_index_id}))", loop_key.c_str(), loop_stride.c_str())},
//             {IndexSymbols::ilp_variable_suffix, "__{ilp_index_id}"}
//         };
//     } else {
//         symbolic_index_initializer = {
//             {IndexSymbols::linear_index, loop_key},
//             {IndexSymbols::ilp_variable_suffix, ""}
//         };
//     }

//     std::function<void(const Op *node)> codegen_dfs = [&](const Op *node) ->
//     void {
//         if (visit.count(node->get_name())) {
//             return;
//         }

//         visit.insert(node->get_name());

//         codege_state_mgr.emit_op(node);

//         for (auto const &operand : node->get_operands()) {
//             codegen_dfs(operand.get());
//         }

//         codege_state_mgr.recall_op(node);

//         cuda_emitter.emit(node, symbolic_index_initializer);
//     };

//     for (auto const &output_node_name : graph->get_output_nodes()) {
//         codegen_dfs(graph->get_node(output_node_name).get());
//     }

//     return cuda_emitter.get_code_stream().str();
// }

void ClusterElewise::setup_codegen() {
  this->get_graph()->build_transitive_closure();

  Loop loop1 = this->get_schedule().get_loop_schedule(0);

  std::string loop1_steps = loop1.get_loop_steps();
  std::string loop1_key = loop1.get_loop_key().get_name();

  // symbolic index to concrete index
  for (auto const& node_name : this->get_graph()->get_node_list()) {
    auto node = this->get_graph()->get_node(node_name);
    node->instantiate_concrete_index({{"linear_index", loop1_key}},
                                     loop1.get_loop_stride());

    if (this->is_instruction_parallelized()) {
      node->instantiate_ilp_concrete_index({{"linear_index", loop1_key}},
                                           loop1.get_loop_ilp_stride(),
                                           loop1.get_loop_stride());
    }
  }

  int loop_element_count =
      this->get_schedule().get_inner_loop().get_loop_shape().element_count();
  int total_thread_count =
      this->cuda_context->cuda_runtime_context.grid_dim.XYZ() *
      this->cuda_context->cuda_runtime_context.block_dim.XYZ();

  if (this->is_instruction_parallelized() &&
      loop_element_count %
              (this->get_instruction_parallel_factor() * total_thread_count) !=
          0) {
    int step2_loop_element_count =
        loop_element_count %
        (this->get_instruction_parallel_factor() * total_thread_count);
    int step1_loop_element_count =
        loop_element_count - step2_loop_element_count;

    Loop existing_loop = this->get_schedule().get_loop_schedule(0);
    std::string step2_loop_init = mononn_engine::helpers::string_format(
        "(%s) + (%d)", existing_loop.get_loop_init().c_str(),
        step1_loop_element_count);

    Loop step1_loop =
        Loop(TensorShape(std::vector<int>{step1_loop_element_count}),
             existing_loop.get_loop_key(), existing_loop.get_loop_init(),
             existing_loop.get_loop_stride());
    step1_loop = step1_loop.instruction_level_parallelism(
        this->get_instruction_parallel_factor());
    Loop step2_loop = Loop(TensorShape(std::vector<int>{loop_element_count}),
                           existing_loop.get_loop_key(), step2_loop_init,
                           existing_loop.get_loop_stride());

    Schedule new_schedule;

    new_schedule.set_locality_tier(this->schedule->get_locality_tier());
    new_schedule.add_loop_schedule(step1_loop);
    new_schedule.add_loop_schedule(step2_loop);

    this->set_schedule(new_schedule);
  }
}

std::string ClusterElewise::generate_cluster_code() const {
  // Do not generate for single constant
  if ((int)this->get_graph()->get_node_list().size() == 1) {
    std::string node_name = this->get_graph()->get_node_list()[0];
    std::shared_ptr<const Op> node = this->get_graph()->get_node(node_name);
    if (node->get_type() == OpType::constant ||
        node->get_type() == OpType::parameter) {
      return "";
    }

    LOG(WARNING) << "Single node cluster, cluster name, "
                 << node->get_cluster_name() << " node name, "
                 << node->get_name();
  }

  std::stringstream ss;

  {  // generate ilp loop
    Loop loop = this->get_schedule().get_loop_schedule(0);
    std::string loop_stride = this->is_instruction_parallelized()
                                  ? loop.get_loop_ilp_stride()
                                  : loop.get_loop_stride();
    std::string loop_boundary = loop.get_loop_condition().get_right_statement();

    std::string loop1_key = loop.get_loop_key().get_name();

    std::vector<std::string> input_nodes = this->get_graph()->get_input_nodes();
    std::vector<std::string> output_nodes =
        this->get_graph()->get_output_nodes();

    ss << loop.begin_loop();

    if (CUDAEmitter::should_use_new_code_emitter(this->graph.get())) {
      ss << CUDAEmitter::emit_code_use_new_code_emitter(
          this->graph.get(), loop.get_loop_key().get_name(),
          loop.get_loop_stride());
    } else {
      this->get_graph()->wave_front_order(
          [&](std::shared_ptr<const Op> node,
              std::shared_ptr<const Op> next_node) -> void {
            std::string node_name = node->get_name();
            std::shared_ptr<OpImplBase> impl = node->get_implementation();

            if (node->need_generate_with_index()) {
              impl->set_need_generate_with_index(true);
              std::string prefetch_pred = mononn_engine::helpers::string_format(
                  "(%s) + (%s) < (%s)", loop1_key.c_str(), loop_stride.c_str(),
                  loop_boundary.c_str());
              impl->propagate_attributes(
                  {{OpAttribute::prefetch_predicate, prefetch_pred}});
            }

            ss << impl->generate();
          });
    }

    ss << loop.end_loop();
  }

  // generate ilp remaining loop
  if (this->is_instruction_parallelized() &&
      this->get_schedule().num_loop_schedule() > 1) {
    // disable ilp for remain loop
    this->graph->set_instruction_parallel_factor(1);
    Loop loop = this->get_schedule().get_loop_schedule(1);
    std::string loop1_key = loop.get_loop_key().get_name();
    std::string loop_stride = loop.get_loop_stride();
    std::string loop_boundary = loop.get_loop_condition().get_right_statement();

    ss << loop.begin_loop();

    if (CUDAEmitter::should_use_new_code_emitter(this->graph.get())) {
      CUDAEmitter::emit_code_use_new_code_emitter(
          this->graph.get(), loop.get_loop_key().get_name(), loop_stride);
    } else {
      this->get_graph()->wave_front_order(
          [&](std::shared_ptr<const Op> node,
              std::shared_ptr<const Op> next_node) -> void {
            std::string node_name = node->get_name();
            std::shared_ptr<OpImplBase> impl = node->get_implementation();

            if (node->need_generate_with_index()) {
              impl->set_need_generate_with_index(true);
              std::string prefetch_pred = mononn_engine::helpers::string_format(
                  "(%s) + (%s) < (%s)", loop1_key.c_str(), loop_stride.c_str(),
                  loop_boundary.c_str());
              impl->propagate_attributes(
                  {{OpAttribute::prefetch_predicate, prefetch_pred}});
            }

            ss << impl->generate();
          });
    }

    this->graph->set_instruction_parallel_factor(
        this->get_instruction_parallel_factor());

    ss << loop.end_loop();
  }

  return ss.str();
}

bool ClusterElewise::is_cluster_elewise() const { return true; }

ClusterType ClusterElewise::get_cluster_type() const {
  return ClusterType::Elewise;
}

std::string ClusterElewise::generate_async_pipeline_initialization() const {
  std::stringstream ss;

  return ss.str();
}

std::string ClusterElewise::generate_async_pipeline_prefetch() const {
  std::stringstream ss;

  return ss.str();
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine