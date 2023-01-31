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

#include "mononn_engine/core/op/cluster_reduce.h"

#include <functional>
#include <sstream>
#include <unordered_map>

#include "mononn_engine/codegen/cuda_emitter.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/core/context/index_tracer.h"
#include "mononn_engine/core/context/index_transform.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/defined.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/gpu/smem_manager.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/pad_impl.h"
#include "mononn_engine/core/op_impl/reduce_impl.h"
#include "mononn_engine/core/op_impl/smem_prefetch_impl.h"
#include "mononn_engine/core/schedule/schedule_factory.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace core {
namespace op {
using Schedule = mononn_engine::core::schedule::Schedule;
using ScheduleFactory = mononn_engine::core::schedule::ScheduleFactory;
using TensorShape = mononn_engine::core::tensor::TensorShape;
using OpType = mononn_engine::core::op::OpType;
using Loop = mononn_engine::core::schedule::Loop;
using Op = mononn_engine::core::op::Op;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using IndexTracer = mononn_engine::core::context::IndexTracer;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using ReduceImpl = mononn_engine::core::op_impl::ReduceImpl;
using Scalar = mononn_engine::core::tensor::Scalar;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using IndexTransform = mononn_engine::core::context::IndexTransform;
using PadImpl = mononn_engine::core::op_impl::PadImpl;
using ClusterType = mononn_engine::core::op_annotation::ClusterType;
using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
using Dtype = mononn_engine::core::tensor::Dtype;
using SmemManager = mononn_engine::core::gpu::SmemManager;
using SmemPrefetchImpl = mononn_engine::core::op_impl::SmemPrefetchImpl;
using Config = mononn_engine::config::Config;
using CUDADefined = mononn_engine::core::gpu::CUDADefined;
using CUDAEmitter = mononn_engine::codegen::CUDAEmitter;

ClusterReduce::ClusterReduce(std::string _name,
                             std::vector<std::shared_ptr<Op>> _operands,
                             std::vector<TensorSpec> _output_specs)
    : ClusterOp(_name, _operands, _output_specs) {
  this->sub_cluster_type_order = {this->get_cluster_type().to_string()};
  this->sub_cluster_tag_order = {this->get_name()};
}

TensorShape ClusterReduce::get_loop_shape() const {
  LOG_ONCE(WARNING, __already_logged, "Need take tuple into account.");

  for (auto const& node_name : this->get_graph()->get_node_list()) {
    std::shared_ptr<const Op> node = this->get_graph()->get_node(node_name);

    if (node->get_type() == OpType::reduce) {
      return node->get_operand(0)->get_output_spec(0).get_shape();
    }
  }

  LOG(FATAL) << "Reduce node not found in cluster " << this->get_name();
}

Schedule ClusterReduce::construct_schedule(LocalityTier::Tier tier) {
  EXPECT_TRUE(tier == LocalityTier::kT1 || tier == LocalityTier::kT2,
              "Unsupported locality tier " + tier.to_string());

  return ScheduleFactory::get_schedule(tier, this->get_loop_shape());
}

void ClusterReduce::trace_symbolic_index() {
  Loop loop2 = this->get_schedule().get_loop_schedule(1);

  // trace symbolic index
  this->get_graph()->reverse_topology_order([&](std::shared_ptr<const Op> node,
                                                std::shared_ptr<const Op>
                                                    next_op) -> void {
    if (this->get_graph()->is_node_traced(node->get_name())) return;

    if (node->get_attribute(OpAttribute::sub_cluster_type) ==
        ClusterType::Reduce.to_string()) {
      this->get_graph()->trace_index("{strided_index}", node->get_name(),
                                     "{linear_index}");
    } else if (node->get_attribute(OpAttribute::sub_cluster_type) ==
               ClusterType::Elewise.to_string()) {
      std::string linear_index_template = mononn_engine::helpers::string_format(
          "(%s * %s) + %s", "{strided_index}",
          std::to_string(loop2.get_loop_shape().element_count()).c_str(),
          "{linear_index}");

      this->get_graph()->trace_index(linear_index_template, node->get_name(),
                                     "");
    } else {
      LOG(FATAL) << "Unsupported initial cluster type: "
                 << node->get_attribute(OpAttribute::sub_cluster_type);
    }
  });
}

void ClusterReduce::setup_codegen() {
  this->get_graph()->build_transitive_closure();

  Loop loop1 = this->get_schedule().get_loop_schedule(0);
  Loop loop2 = this->get_schedule().get_loop_schedule(1);

  std::string loop1_key = loop1.get_loop_key().get_name();
  std::string loop2_key = loop2.get_loop_key().get_name();

  // symbolic index to concrete index
  for (auto const& node_name : this->get_graph()->get_node_list()) {
    auto node = this->get_graph()->get_node(node_name);
    std::map<std::string, std::string> params = {{"strided_index", loop1_key},
                                                 {"linear_index", loop2_key}};

    if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
      if (this->get_schedule().get_locality_tier() == LocalityTier::kT1) {
        params["pipeline_initialization_strided_index"] =
            mononn_engine::helpers::string_format(
                "(%s + stage_id * %s)", CUDADefined::warp_global_id.c_str(),
                CUDADefined::warp_global_count.c_str());

        params["pipeline_prefetch_strided_index"] =
            mononn_engine::helpers::string_format(
                "(%s + (%s - 1) * %s)", loop1_key.c_str(),
                this->async_pipeline_stage_count_codegen_var_name.c_str(),
                CUDADefined::warp_global_count.c_str());
      } else if (this->get_schedule().get_locality_tier() ==
                 LocalityTier::kT2) {
        params["pipeline_initialization_strided_index"] =
            mononn_engine::helpers::string_format(
                "(%s + stage_id * %s)", CUDADefined::blockIdx_x.c_str(),
                CUDADefined::gridDim_x.c_str());

        params["pipeline_prefetch_strided_index"] =
            mononn_engine::helpers::string_format(
                "(%s + (%s - 1) * %s)", loop1_key.c_str(),
                this->async_pipeline_stage_count_codegen_var_name.c_str(),
                CUDADefined::gridDim_x.c_str());
      } else {
        LOG(FATAL) << "Unsupported locality tier: "
                   << this->get_schedule().get_locality_tier().to_string();
      }

      params["pipeline_initialization_linear_index"] = loop2_key;
      params["pipeline_prefetch_linear_index"] = loop2_key;
      // params["smem_access_index"] = loop2_key;
      params["prefetch_loop_boundary"] =
          std::to_string(loop1.get_loop_shape().element_count());
      params["async_pipeline_stage_count_codegen_var_name"] =
          this->async_pipeline_stage_count_codegen_var_name;
    }

    node->instantiate_concrete_index(params, loop2.get_loop_stride());

    if (this->is_instruction_parallelized()) {
      node->instantiate_ilp_concrete_index(params, loop2.get_loop_ilp_stride(),
                                           loop2.get_loop_stride());
    }
  }

  // Set loop for ILP remaining loop (if any)
  int inner_loop_element_count = loop2.get_loop_shape().element_count();
  int cooperate_thread_count;  // Number of threads cooperate in reduce cluster
  if (this->get_schedule().get_locality_tier() == LocalityTier::kT1) {
    cooperate_thread_count = this->cuda_context->cuda_device_context.warp_size;
  } else if (this->get_schedule().get_locality_tier() == LocalityTier::kT2) {
    cooperate_thread_count =
        this->cuda_context->cuda_runtime_context.block_dim.XYZ();
  } else {
    LOG(FATAL) << "";
  }

  if (this->is_instruction_parallelized() &&
      (inner_loop_element_count % (this->get_instruction_parallel_factor() *
                                   cooperate_thread_count)) != 0) {
    int step2_loop_element_count =
        inner_loop_element_count %
        (this->get_instruction_parallel_factor() * cooperate_thread_count);
    int step1_loop_element_count =
        inner_loop_element_count - step2_loop_element_count;

    Loop existing_loop = this->get_schedule().get_loop_schedule(1);
    std::string step2_loop_init = mononn_engine::helpers::string_format(
        "(%s) + (%d)", existing_loop.get_loop_init().c_str(),
        step1_loop_element_count);

    Loop step1_inner_loop =
        Loop(TensorShape(std::vector<int>{step1_loop_element_count}),
             existing_loop.get_loop_key(), existing_loop.get_loop_init(),
             existing_loop.get_loop_stride());
    step1_inner_loop = step1_inner_loop.instruction_level_parallelism(
        this->get_instruction_parallel_factor());
    Loop step2_inner_loop =
        Loop(TensorShape(std::vector<int>{inner_loop_element_count}),
             existing_loop.get_loop_key(), step2_loop_init,
             existing_loop.get_loop_stride());

    Schedule new_schedule;
    new_schedule.add_loop_schedule(this->get_schedule().get_loop_schedule(0));
    new_schedule.add_loop_schedule(step1_inner_loop);
    new_schedule.add_loop_schedule(step2_inner_loop);

    new_schedule.set_locality_tier(this->schedule->get_locality_tier());

    this->set_schedule(new_schedule);
  }

  for (auto const& node_name : this->graph->get_node_list()) {
    auto node = this->graph->get_node(node_name);

    // set attribute for cache prefetch
    if (node->has_attribute(OpAttribute::is_parameter_cache_prefetched)) {
      std::string loop2_stride = this->is_instruction_parallelized()
                                     ? loop2.get_loop_ilp_stride()
                                     : loop2.get_loop_stride();
      std::string loop2_boundary =
          loop2.get_loop_condition().get_right_statement();
      std::string prefetch_pred = mononn_engine::helpers::string_format(
          "(%s) + (%s) < (%s)", loop2_key.c_str(), loop2_stride.c_str(),
          loop2_boundary.c_str());
      node->get_implementation()->propagate_attributes(
          {{OpAttribute::prefetch_predicate, prefetch_pred}});
    }
  }
}

std::string ClusterReduce::generate_cluster_code() const {
  Loop loop1 = this->get_schedule().get_loop_schedule(0);
  Loop loop2 = this->get_schedule().get_loop_schedule(1);
  std::string loop2_steps = loop2.get_loop_steps();

  std::stringstream ss;

  ss << this->smem_manager->define_root_buffer();

  if (this->schedule->get_locality_tier() == LocalityTier::kT2) {
    ss << "void *" << Config::get()->smem_reduction_cache_name << " = "
       << this->smem_manager->get_buffer_pointer(
              Config::get()->smem_reduction_cache_name)
       << ";\n";
  }

  if (this->is_cluster_contain_async_prefetched_nodes()) {
    ss << this->generate_async_pipeline_initialization();
  }

  ss << loop1.begin_loop();

  if (this->is_cluster_contain_async_prefetched_nodes()) {
    ss << this->generate_async_pipeline_prefetch();
  }

  std::vector<std::string> output_nodes = this->get_graph()->get_output_nodes();
  std::vector<std::string> input_nodes = this->get_graph()->get_input_nodes();

  std::string current_sub_cluster_tag = "";
  std::string current_sub_cluster_type = "";
  std::vector<std::string> codegen_reduce_group;  // group of reduce node for
                                                  // together code generation
  std::vector<std::string> current_sub_cluster_node_list;

  std::function<void()> generate_ilp_remaining_loop = [&]() -> void {
    int current_ilp_factor = this->get_instruction_parallel_factor();
    this->graph->set_instruction_parallel_factor(1);

    Loop loop2_remaining_loop = this->get_schedule().get_loop_schedule(2);

    ss << loop2_remaining_loop.begin_loop();

    if (CUDAEmitter::should_use_new_code_emitter(this->graph.get())) {
      ss << CUDAEmitter::emit_code_use_new_code_emitter(
          this->graph.get(), loop2_remaining_loop.get_loop_key().get_name(),
          loop2_remaining_loop.get_loop_stride(),
          loop1.get_loop_key().get_name());
    } else {
      for (auto const& node_name : current_sub_cluster_node_list) {
        std::shared_ptr<Op> node_remaining = this->graph->get_node(node_name);
        std::shared_ptr<OpImplBase> impl_remaining =
            node_remaining->get_implementation();

        if (node_remaining->need_generate_with_index()) {
          impl_remaining->set_need_generate_with_index(true);
          std::string loop2_key =
              loop2_remaining_loop.get_loop_key().get_name();
          std::string loop2_stride = loop2_remaining_loop.get_loop_stride();
          std::string loop2_boundary =
              loop2_remaining_loop.get_loop_condition().get_right_statement();
          std::string prefetch_pred = mononn_engine::helpers::string_format(
              "(%s) + (%s) < (%s)", loop2_key.c_str(), loop2_stride.c_str(),
              loop2_boundary.c_str());
          impl_remaining->propagate_attributes(
              {{OpAttribute::prefetch_predicate, prefetch_pred}});
        }

        ss << impl_remaining->generate();
      }

      ss << loop2_remaining_loop.end_loop();
    }

    this->graph->set_instruction_parallel_factor(current_ilp_factor);
  };

  if (CUDAEmitter::should_use_new_code_emitter(this->graph.get())) {
    std::vector<std::string> pre_inner_loop_nodes =
        this->get_graph()->get_node([&](const Op* op2) -> bool {
          return op2->need_pre_inner_loop_generation();
        });

    // pre inner loop
    for (auto const node_name : pre_inner_loop_nodes) {
      ss << this->get_graph()->get_node(node_name)->generate_pre_inner_loop();
    }

    ss << loop2.begin_loop();
    ss << CUDAEmitter::emit_code_use_new_code_emitter(
        this->graph.get(), loop2.get_loop_key().get_name(),
        loop2.get_loop_stride(), loop1.get_loop_key().get_name());
  } else {
    this->get_graph()->wave_front_order([&](std::shared_ptr<const Op> op,
                                            std::shared_ptr<const Op> next_op)
                                            -> void {
      // step into a new subcluster
      if (op->get_attribute(OpAttribute::sub_cluster_tag) !=
          current_sub_cluster_tag) {
        current_sub_cluster_tag =
            op->get_attribute(OpAttribute::sub_cluster_tag);
        current_sub_cluster_type =
            op->get_attribute(OpAttribute::sub_cluster_type);

        ss << "// Enter sub-cluster: " << current_sub_cluster_tag
           << " sub-cluster type: " << current_sub_cluster_type << "\n";

        current_sub_cluster_node_list.clear();

        std::vector<std::string> pre_inner_loop_nodes =
            this->get_graph()->get_node([&](const Op* op2) -> bool {
              return op2->get_attribute(OpAttribute::sub_cluster_tag) ==
                         current_sub_cluster_tag &&
                     op2->need_pre_inner_loop_generation();
            });

        std::vector<std::string> post_inner_loop_nodes =
            this->get_graph()->get_node([&](const Op* op2) -> bool {
              return op2->get_attribute(OpAttribute::sub_cluster_tag) ==
                         current_sub_cluster_tag &&
                     op2->need_post_inner_loop_generation();
            });

        // pre inner loop
        for (auto const node_name : pre_inner_loop_nodes) {
          ss << this->get_graph()
                    ->get_node(node_name)
                    ->generate_pre_inner_loop();
        }

        ss << loop2.begin_loop();

        // post inner loop
        for (auto const& node_name : post_inner_loop_nodes) {
          ss << this->get_graph()
                    ->get_node(node_name)
                    ->generate_post_inner_loop();
        }
      }

      current_sub_cluster_node_list.push_back(op->get_name());

      std::shared_ptr<OpImplBase> impl = op->get_implementation();

      if (op->need_generate_with_index()) {
        impl->set_need_generate_with_index(true);
      }

      ss << impl->generate();

      // end of a sub-cluster
      if (next_op == nullptr ||
          next_op->get_attribute(OpAttribute::sub_cluster_tag) !=
              current_sub_cluster_tag) {
        ss << loop2.end_loop();
        if (this->schedule->num_loop_schedule() > 2) {
          // generate ILP remaining loop
          generate_ilp_remaining_loop();
        }

        ss << "// End of sub-cluster: " << current_sub_cluster_tag << "\n";
      }

      if (op->get_type() == OpType::reduce) {
        codegen_reduce_group.push_back(op->get_name());

        if (next_op != nullptr &&
            next_op->get_attribute(OpAttribute::sub_cluster_tag) ==
                current_sub_cluster_tag) {
          // not reaching end of a sub-cluster
          return;
        }

        for (auto const& reduce_node_name : codegen_reduce_group) {
          auto reduce_node = this->graph->get_node(reduce_node_name);
          auto reduce_impl = reduce_node->get_implementation();
          ss << reduce_impl->as<ReduceImpl>()->generate_reduce() << "\n";
        }

        bool generated_post_reduce_if = false;

        for (auto const& reduce_node_name : codegen_reduce_group) {
          if (std::find(output_nodes.begin(), output_nodes.end(),
                        reduce_node_name) == output_nodes.end()) {
            continue;
          }

          auto reduce_node = this->graph->get_node(reduce_node_name);
          auto reduce_impl = reduce_node->get_implementation();

          if (!generated_post_reduce_if) {
            generated_post_reduce_if = true;
            ss << reduce_impl->as<ReduceImpl>()->get_post_reduce_if_statement();
          }

          std::vector<ConcreteIndexStamp> concrete_index =
              reduce_impl->get_concrete_index();
          EXPECT_TRUE(concrete_index.size() == 1,
                      "Should have only one output index");
          if (reduce_impl->get_reduce_accum().is_tuple()) {
            std::vector<Dtype> tuple_type_list =
                reduce_impl->get_reduce_accum().get_types_in_list();

            for (int tuple_idx = 0; tuple_idx < tuple_type_list.size();
                 ++tuple_idx) {
              std::string node_buffer_name =
                  mononn_engine::helpers::string_format(
                      "%s_tuple_index_%d_output",
                      BufferManager::get_buffer_name(reduce_node_name).c_str(),
                      tuple_idx);
              ss << Memory::write(Memory::AccessFlavor::REGULAR,
                                  op->get_output_spec(tuple_idx).get_dtype(),
                                  mononn_engine::helpers::string_format(
                                      "cuda::std::get<%d>(%s)", tuple_idx,
                                      reduce_node_name.c_str()),
                                  node_buffer_name,
                                  concrete_index[0].index_before_trace);
            }
          } else {
            std::string node_buffer_name =
                BufferManager::get_buffer_name(reduce_node_name) + "_output";
            ss << Memory::write(Memory::AccessFlavor::REGULAR,
                                op->get_output_spec(0).get_dtype(),
                                reduce_node_name, node_buffer_name,
                                concrete_index[0].index_before_trace);
          }
        }

        if (generated_post_reduce_if) {
          ss << impl->as<ReduceImpl>()->get_post_reduce_if_end() << "\n\n";
        }

        codegen_reduce_group.clear();
      }
    });
  }

  if (this->is_cluster_contain_async_prefetched_nodes()) {
    ss << this->generate_async_pipeline_stage_increment();
  }

  ss << loop1.end_loop();

  if (this->is_cluster_contain_async_prefetched_nodes()) {
    ss << this->generate_async_pipeline_flush();
  }

  return ss.str();
}

std::vector<Op*> ClusterReduce::get_reduce_nodes() {
  std::vector<Op*> reduce_node_list;
  for (auto const& node_name :
       this->get_graph()->get_node_list_by_type(OpType::reduce)) {
    std::shared_ptr<Op> node = this->get_graph()->get_node(node_name);
    reduce_node_list.push_back(node.get());
  }

  if (reduce_node_list.empty())
    LOG(FATAL) << "Reduce node not found in reduce cluster";

  return reduce_node_list;
}

std::vector<const Op*> ClusterReduce::get_reduce_nodes() const {
  std::vector<const Op*> reduce_node_list;
  for (auto const& node_name :
       this->get_graph()->get_node_list_by_type(OpType::reduce)) {
    std::shared_ptr<const Op> node = this->get_graph()->get_node(node_name);
    reduce_node_list.push_back(node.get());
  }

  if (reduce_node_list.empty())
    LOG(FATAL) << "Reduce node not found in reduce cluster";

  return reduce_node_list;
}

std::vector<Op*> ClusterReduce::get_reduce_nodes_in_last_sub_cluster() {
  std::vector<std::string> candidates =
      this->get_graph()->get_node_list_by_type(OpType::reduce);

  EXPECT_TRUE(candidates.size() != 0, "No reduce node found");

  std::string result = candidates[0];

  for (int idx = 1; idx < (int)candidates.size(); ++idx) {
    if (this->get_graph()->topology_before(result, candidates[idx])) {
      result = candidates[idx];
    } else {
      if (this->get_graph()->get_node_attribute(candidates[idx],
                                                OpAttribute::sub_cluster_tag) !=
          this->get_graph()->get_node_attribute(result,
                                                OpAttribute::sub_cluster_tag)) {
        EXPECT_TRUE(this->get_graph()->topology_before(candidates[idx], result),
                    "Reduce node " + result + " and " + candidates[idx] +
                        " not in same sub cluster and have no topology "
                        "relationship in cluster " +
                        this->get_name());
      }
    }
  }

  std::vector<Op*> reduce_node_list;

  for (auto const& node_name : candidates) {
    if (this->get_graph()->get_node_attribute(node_name,
                                              OpAttribute::sub_cluster_tag) ==
        this->get_graph()->get_node_attribute(result,
                                              OpAttribute::sub_cluster_tag)) {
      reduce_node_list.push_back(this->get_graph()->get_node(node_name).get());
    }
  }

  return reduce_node_list;
}

std::vector<const Op*> ClusterReduce::get_reduce_nodes_in_last_sub_cluster()
    const {
  std::vector<std::string> candidates =
      this->get_graph()->get_node_list_by_type(OpType::reduce);

  EXPECT_TRUE(candidates.size() != 0, "No reduce node found");

  std::string result = candidates[0];

  for (int idx = 1; idx < (int)candidates.size(); ++idx) {
    if (this->get_graph()->topology_before(result, candidates[idx])) {
      result = candidates[idx];
    } else {
      if (this->get_graph()->get_node_attribute(candidates[idx],
                                                OpAttribute::sub_cluster_tag) !=
          this->get_graph()->get_node_attribute(result,
                                                OpAttribute::sub_cluster_tag)) {
        EXPECT_TRUE(this->get_graph()->topology_before(candidates[idx], result),
                    "Reduce node " + result + " and " + candidates[idx] +
                        " not in same sub cluster and have no topology "
                        "relationship in cluster " +
                        this->get_name());
      }
    }
  }

  std::vector<const Op*> reduce_node_list;

  for (auto const& node_name : candidates) {
    if (this->get_graph()->get_node_attribute(node_name,
                                              OpAttribute::sub_cluster_tag) ==
        this->get_graph()->get_node_attribute(result,
                                              OpAttribute::sub_cluster_tag)) {
      reduce_node_list.push_back(this->get_graph()->get_node(node_name).get());
    }
  }

  return reduce_node_list;
}

bool ClusterReduce::is_cluster_reduce() const { return true; }

ClusterType ClusterReduce::get_cluster_type() const {
  return ClusterType::Reduce;
}

int ClusterReduce::get_reduction_dimension_size() const {
  return this->get_reduce_nodes()[0]
      ->get_operand(0)
      ->get_output_spec(0)
      .get_shape(-1);
}

std::string ClusterReduce::generate_async_pipeline_initialization() const {
  std::stringstream ss;
  Loop inner_most_loop = this->get_schedule().get_inner_loop();

  // Generate async pipeline initialization.
  ss << "// Async pipeline setup.\n";
  ss << mononn_engine::helpers::string_format(
      "constexpr int %s = %s;\n",
      this->async_pipeline_stage_count_codegen_var_name.c_str(),
      this->get_attribute(OpAttribute::async_pipeline_total_stage_count)
          .c_str());
  for (auto const& node_name :
       this->graph->get_node_list_by_type(OpType::parameter)) {
    auto node = this->graph->get_node(node_name);

    if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
      if (node->get_symbolic_index().size() > 1) {  // traced by multiple nodes.
        for (auto const& symbolic_index : node->get_symbolic_index()) {
          std::string reuse_node_name =
              node_name + "_reuse_" + symbolic_index.traced_by;
          std::string smem_buf_name =
              this->smem_manager->get_buffer_name(reuse_node_name);
          std::string smem_ptr_type;

          Dtype type = node->get_output_spec(0).get_dtype();

          ss << type.to_string() << " "
             << mononn_engine::helpers::string_format("(*%s)",
                                                      smem_buf_name.c_str());
          smem_ptr_type += type.to_string() + " (*)";

          if (this->get_schedule().get_locality_tier() == LocalityTier::kT1) {
            ss << mononn_engine::helpers::string_format(
                "[%s]",
                this->async_pipeline_stage_count_codegen_var_name.c_str());
            smem_ptr_type += mononn_engine::helpers::string_format(
                "[%s]",
                this->async_pipeline_stage_count_codegen_var_name.c_str());
          }

          // int per_warp_or_block_per_stage_buffer_size_in_bytes =
          // type.size_in_bytes() *
          // inner_most_loop.get_loop_shape().element_count();
          size_t base_buf_size_in_bytes =
              this->smem_manager->get_base_buffer_size(smem_buf_name) /
              type.size_in_bytes();
          ss << mononn_engine::helpers::string_format(
              "[%d]", (int)base_buf_size_in_bytes);
          smem_ptr_type += mononn_engine::helpers::string_format(
              "[%d]", (int)base_buf_size_in_bytes);

          ss << " = "
             << this->smem_manager->get_buffer_pointer(smem_buf_name,
                                                       smem_ptr_type)
             << ";\n";
        }
      } else {
        std::string smem_buf_name =
            this->smem_manager->get_buffer_name(node_name);
        std::string smem_ptr_type;

        Dtype type = node->get_output_spec(0).get_dtype();

        ss << type.to_string() << " "
           << mononn_engine::helpers::string_format("(*%s)",
                                                    smem_buf_name.c_str());
        smem_ptr_type += type.to_string() + " (*)";

        if (this->get_schedule().get_locality_tier() == LocalityTier::kT1) {
          ss << mononn_engine::helpers::string_format(
              "[%s]",
              this->async_pipeline_stage_count_codegen_var_name.c_str());
          smem_ptr_type += mononn_engine::helpers::string_format(
              "[%s]",
              this->async_pipeline_stage_count_codegen_var_name.c_str());
        }

        size_t base_buf_size_in_bytes =
            this->smem_manager->get_base_buffer_size(smem_buf_name) /
            type.size_in_bytes();
        // int per_warp_or_block_per_stage_buffer_size_in_bytes =
        // type.size_in_bytes() *
        // inner_most_loop.get_loop_shape().element_count();
        ss << mononn_engine::helpers::string_format(
            "[%d]", (int)base_buf_size_in_bytes);
        smem_ptr_type += mononn_engine::helpers::string_format(
            "[%d]", (int)base_buf_size_in_bytes);

        ss << " = "
           << this->smem_manager->get_buffer_pointer(smem_buf_name,
                                                     smem_ptr_type)
           << ";\n";
      }
    }
  }

  Loop reduce_inner_loop = this->get_schedule().get_loop_schedule(1);

  ss << "#pragma unroll\n";
  ss << mononn_engine::helpers::string_format(
      "for (int stage_id = 0; stage_id <= %s - 2; ++stage_id) {\n",
      this->async_pipeline_stage_count_codegen_var_name.c_str());
  ss << reduce_inner_loop.begin_loop();

  for (auto const& node_name : this->graph->traverse_in_topology_order()) {
    auto node = this->graph->get_node(node_name);
    // TODO deal with ilp remaining loop
    if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
      ss << node->get_implementation()
                ->as<SmemPrefetchImpl>()
                ->generate_async_pipeline_initialization();
    }
  }

  ss << reduce_inner_loop.end_loop();

  if (this->get_schedule().num_loop_schedule() >
      2) {  // have ilp remaining loop
    Loop reduce_inner_remaining_loop =
        this->get_schedule().get_loop_schedule(2);

    ss << reduce_inner_remaining_loop.begin_loop();

    int current_ilp_factor = this->get_instruction_parallel_factor();

    this->graph->set_instruction_parallel_factor(1);
    for (auto const& node_name : this->graph->traverse_in_topology_order()) {
      auto node = this->graph->get_node(node_name);
      if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
        ss << node->get_implementation()
                  ->as<SmemPrefetchImpl>()
                  ->generate_async_pipeline_initialization();
      }
    }

    ss << reduce_inner_remaining_loop.end_loop();

    this->graph->set_instruction_parallel_factor(current_ilp_factor);
  }

  ss << "asynchronous::commit()();\n";
  ss << "}\n";
  ss << mononn_engine::helpers::string_format(
      "int stage_id = 0;\n",
      this->async_pipeline_stage_count_codegen_var_name.c_str());
  ss << "// Async pipeline setup end.\n";

  return ss.str();
}

std::string ClusterReduce::generate_async_pipeline_prefetch() const {
  std::stringstream ss;
  Loop reduce_inner_loop = this->get_schedule().get_loop_schedule(1);

  ss << reduce_inner_loop.begin_loop();

  for (auto const& node_name : this->graph->traverse_in_topology_order()) {
    auto node = this->graph->get_node(node_name);
    // todo deal with ilp remaining loop
    if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
      ss << node->get_implementation()
                ->as<SmemPrefetchImpl>()
                ->generate_async_pipeline_prefetch();
    }
  }

  ss << reduce_inner_loop.end_loop();

  if (this->get_schedule().num_loop_schedule() > 2) {
    Loop reduce_inner_remaining_loop =
        this->get_schedule().get_loop_schedule(2);

    ss << reduce_inner_remaining_loop.begin_loop();

    int current_ilp_factor = this->get_instruction_parallel_factor();
    this->graph->set_instruction_parallel_factor(1);
    for (auto const& node_name : this->graph->traverse_in_topology_order()) {
      auto node = this->graph->get_node(node_name);
      // todo deal with ilp remaining loop
      if (node->has_attribute(OpAttribute::is_parameter_async_prefetched)) {
        ss << node->get_implementation()
                  ->as<SmemPrefetchImpl>()
                  ->generate_async_pipeline_prefetch();
      }
    }

    ss << reduce_inner_remaining_loop.end_loop();
    this->graph->set_instruction_parallel_factor(current_ilp_factor);
  }

  ss << "asynchronous::commit()();\n";
  ss << mononn_engine::helpers::string_format(
      "asynchronous::wait<%s - 1>()();\n",
      this->async_pipeline_stage_count_codegen_var_name.c_str());
  // ss << "__syncthreads();\n";
  return ss.str();
}

void ClusterReduce::initialize_smem_manager() {
  if (!this->schedule) {
    LOG(FATAL) << "Cannot initialize smem manager until schedule assigned.";
  }

  auto reduce_nodes = this->get_reduce_nodes();

  for (auto node : reduce_nodes) {
    Dtype type = node->get_output_spec(0).get_dtype();

    if (type != Dtype::FLOAT16 && type != Dtype::FLOAT32 &&
        type != Dtype::INT32 && type != Dtype::BOOL) {
      LOG(FATAL) << "Unsupported type: " << type.to_string() << " in node "
                 << node->get_name();
    }
  }

  this->smem_manager = std::make_shared<SmemManager>(
      this->cuda_context->cuda_runtime_context.smem_size);

  if (this->schedule->get_locality_tier() == LocalityTier::kT2) {
    this->smem_manager->claim_smem_buffer(
        Config::get()->smem_reduction_cache_name, 640);
  }
}
}  // namespace op
}  // namespace core
}  // namespace mononn_engine