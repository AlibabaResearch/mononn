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

#include "mononn_engine/codegen/cuda_emitter.h"

#include <map>

#include "mononn_engine/core/context/index_tracer.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/gpu/memory.h"
#include "mononn_engine/core/op/broadcast.h"
#include "mononn_engine/core/op/compare.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/convert.h"
#include "mononn_engine/core/op/gather.h"
#include "mononn_engine/core/op/multiply.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/parameter.h"
#include "mononn_engine/core/op/reduce_window.h"
#include "mononn_engine/core/op_impl//reduce_impl.h"
#include "mononn_engine/core/semantic/function_invocation.h"
#include "mononn_engine/helpers/stl_helpers.h"

namespace mononn_engine {
namespace codegen {
using OpType = mononn_engine::core::op::OpType;
using Broadcast = mononn_engine::core::op::Broadcast;
using Convert = mononn_engine::core::op::Convert;
using Constant = mononn_engine::core::op::Constant;
using Compare = mononn_engine::core::op::Compare;
using Functor = mononn_engine::core::gpu::Functor;
using ReduceWindow = mononn_engine::core::op::ReduceWindow;
using FunctionInvocation = mononn_engine::core::semantic::FunctionInvocation;
using Memory = mononn_engine::core::gpu::Memory;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using ReduceImpl = mononn_engine::core::op_impl::ReduceImpl;
using Dtype = mononn_engine::core::tensor::Dtype;

bool CUDAEmitter::should_use_new_code_emitter(const Graph* graph) {
  for (auto const& node_name : graph->get_node_list()) {
    auto node = graph->get_node(node_name);

    if (node->get_type() == OpType::reduce_window) {
      return true;
    }
  }

  return false;
}

std::string CUDAEmitter::emit_code_use_new_code_emitter(
    const Graph* graph, const std::string& linear_loop_key,
    const std::string& linear_loop_stride,
    const std::string& strided_loop_key) {
  using CUDAEmitter = mononn_engine::codegen::CUDAEmitter;
  using CodegenStateSpaceMgr = mononn_engine::codegen::CodegenStateSpaceMgr;
  using IndexSymbols = mononn_engine::core::context::IndexSymbols;

  CodegenStateSpaceMgr codege_state_mgr;
  CUDAEmitter cuda_emitter(graph, codege_state_mgr.get_codegen_state_space());
  std::set<std::pair<std::string, std::string>>
      visit;  // pair<node_name, traced_by>;

  std::map<std::string, std::string> symbolic_index_initializer;

  bool has_reduce_node = !graph->get_node_list_by_type(OpType::reduce).empty();

  if (graph->get_instruction_parallel_factor() != 1) {
    codege_state_mgr.emit_instruction_level_parallelism(
        graph->get_instruction_parallel_factor());
    symbolic_index_initializer = {
        {IndexSymbols::linear_index,
         mononn_engine::helpers::string_format("((%s) + (%s * {ilp_index_id}))",
                                               linear_loop_key.c_str(),
                                               linear_loop_stride.c_str())},
        {IndexSymbols::ilp_variable_suffix, "__i{ilp_index_id}"},
        {IndexSymbols::strided_index, strided_loop_key}};
  } else {
    symbolic_index_initializer = {
        {IndexSymbols::linear_index, linear_loop_key},
        {IndexSymbols::ilp_variable_suffix, ""},
        {IndexSymbols::strided_index, strided_loop_key}};
  }

  std::function<void(const Op*, const std::string&, const Op*)> codegen_dfs =
      [&](const Op* node, const std::string& source_index,
          const Op* source_node) -> void {
    auto visit_mark = std::make_pair(node->get_name(), source_index);

    if (visit.count(visit_mark)) {
      return;
    }

    for (auto const& symbolic_index : node->get_symbolic_index()) {
      codege_state_mgr.emit_op(node);
      if (symbolic_index.index_before_trace == source_index ||
          (source_node &&
           source_node->get_type() == OpType::gather)) {  // This is a hack
        visit.insert(visit_mark);

        // This reverse ordered operand traversal is a trick.
        // Need a better way to efficiently manage DFS context while respect
        // control dependency. Just reverse the order of operand here is only
        // gather operator will introduce control dependency.
        for (int operand_id = node->get_operand_count() - 1; operand_id >= 0;
             --operand_id) {
          codegen_dfs(node->get_operand(operand_id).get(),
                      symbolic_index.index_after_trace, node);
        }

        // for (auto const &control_edge :
        // graph->get_node_input_control_edges(node->get_name())) {
        //     codegen_dfs(control_edge->get_src().get(),
        //     symbolic_index.index_after_trace, node);
        // }

        codege_state_mgr.recall_op(node);

        cuda_emitter.emit(
            node, symbolic_index.instantiate(symbolic_index_initializer));
      }
    }

    if (node->get_type() == OpType::reduce) {
      cuda_emitter.emit_loop_end(node);

      cuda_emitter.emit_reduce_broadcast(node);
      cuda_emitter.emit_post_reduce_if(node);

      for (auto const& symbolic_index : node->get_symbolic_index()) {
        cuda_emitter.emit_output(
            node, symbolic_index.instantiate(symbolic_index_initializer));
      }

      cuda_emitter.emit_post_reduce_if_end(node);
    }
  };

  for (auto const& output_node_name : graph->get_output_nodes()) {
    auto output_node = graph->get_node(output_node_name);

    for (auto const& symbolic_index : output_node->get_symbolic_index()) {
      codegen_dfs(graph->get_node(output_node_name).get(),
                  symbolic_index.index_before_trace, nullptr);
    }
  }

  return cuda_emitter.get_code_stream().str();
}

void CUDAEmitter::emit(const Op* node,
                       const ConcreteIndexStamp& concrete_index) {
  CodegenState codegen_state;
  codegen_state.node_name_suffix = "";
  codegen_state.index = concrete_index;

  if (node->get_type() == OpType::reduce) {
    this->emit_codegen_state(node, codegen_state);
  } else {
    std::vector<CodegenState> concrete_codegen_state_list =
        this->codegens_state_space->generate_codegen_state(codegen_state);

    for (auto const& state : concrete_codegen_state_list) {
      this->emit_codegen_state(node, state);
    }
  }
}

// void CUDAEmitter::emit(const Op *node, const std::map<std::string,
// std::string> &symbolic_index_initializer) {
//     std::vector<CodegenState> symbolic_codegen_state_list;

//     for (auto const index : node->get_symbolic_index()) {
//         CodegenState codegen_state;
//         codegen_state.node_name_suffix = "";
//         codegen_state.index = index.instantiate(symbolic_index_initializer);
//         symbolic_codegen_state_list.push_back(codegen_state);
//     }

//     LOG(ERROR) << "Generate codegen state for node " << node->get_name();
//     std::vector<CodegenState> concrete_codegen_state_list
//         =
//         this->codegens_state_space->generate_codegen_state(symbolic_codegen_state_list);

//     for (auto const &state : concrete_codegen_state_list) {
//         this->emit_codegen_state(node, state);
//     }
// }

void CUDAEmitter::emit_codegen_state(const Op* node,
                                     const CodegenState& codegen_state) {
  if (node->get_type() == OpType::broadcast ||
      node->get_type() == OpType::gather) {
    this->emit_non_op(node, codegen_state);
  } else if (node->get_type() == OpType::convert) {
    this->emit_elementwise_unary(node, codegen_state);
  } else if (node->get_type() == OpType::constant) {
    this->emit_constant(node, codegen_state);
  } else if (node->get_type() == OpType::compare ||
             node->get_type() == OpType::multiply ||
             node->get_type() == OpType::add) {
    this->emit_elementwise_binary(node, codegen_state);
  } else if (node->get_type() == OpType::parameter) {
    this->emit_parameter(node, codegen_state);
  } else if (node->get_type() == OpType::reduce) {
    this->emit_reduce(node, codegen_state);
  } else if (node->get_type() == OpType::reduce_window) {
    this->emit_reduce_window(node, codegen_state);
  } else if (node->get_type() == OpType::select) {
    this->emit_select(node, codegen_state);
  } else {
    LOG(FATAL) << node->get_name() << " have unsupported type "
               << node->get_type().to_string();
  }

  if (graph->is_output_node(node->get_name()) &&
      node->get_type() != OpType::reduce) {
    this->emit_output(node, codegen_state);
  }
}

void CUDAEmitter::emit_constant(const Op* node,
                                const CodegenState& codegen_state) {
  if (node->as<Constant>()->is_scalar()) {
    this->code_stream << mononn_engine::helpers::string_format(
        "const %s %s = %s;\n",
        node->get_output_spec(0).get_dtype().to_string().c_str(),
        (node->get_name() + codegen_state.node_name_suffix).c_str(),
        node->as<Constant>()->get_value().c_str());
  }
}

void CUDAEmitter::emit_elementwise_unary(const Op* node,
                                         const CodegenState& codegen_state) {
  // Convert
  std::string functor_name =
      Functor::get_functor_name_for_op_type(node->get_type());
  Functor functor(functor_name, node->get_output_spec(0).get_dtype());
  FunctionInvocation invocation(functor.get_name());
  invocation.add_arg(node->get_operand(0)->get_name() +
                     codegen_state.node_name_suffix);

  // this->code_stream << invocation.to_string();
  this->code_stream << mononn_engine::helpers::string_format(
      "%s %s = %s;\n", node->get_output_spec(0).get_dtype().to_string().c_str(),
      (node->get_name() + codegen_state.node_name_suffix).c_str(),
      invocation.to_string().c_str());
}

void CUDAEmitter::emit_elementwise_binary(const Op* node,
                                          const CodegenState& codegen_state) {
  // Compare Multiply
  std::string functor_name;

  if (node->get_type() == OpType::compare) {
    functor_name = Functor::get_functor_name_for_op_type(
        OpType::compare, node->as<Compare>()->get_comparator());
  } else {
    functor_name = Functor::get_functor_name_for_op_type(node->get_type());
  }

  Functor functor(functor_name,
                  node->get_operand(0)->get_output_spec(0).get_dtype());
  FunctionInvocation invocation(functor.get_name());

  invocation.add_arg(node->get_operand(0)->get_name() +
                     codegen_state.node_name_suffix);
  invocation.add_arg(node->get_operand(1)->get_name() +
                     codegen_state.node_name_suffix);

  this->code_stream << mononn_engine::helpers::string_format(
      "%s %s = %s;\n", node->get_output_spec(0).get_dtype().to_string().c_str(),
      (node->get_name() + codegen_state.node_name_suffix).c_str(),
      invocation.to_string().c_str());
}

void CUDAEmitter::emit_parameter(const Op* node,
                                 const CodegenState& codegen_state) {
  std::string var_name = node->get_name() + codegen_state.node_name_suffix;
  std::string buffer_name =
      BufferManager::get_buffer_name(node->get_name()) + "_input";
  auto dtype = node->get_output_spec(0).get_dtype();
  this->code_stream << Memory::read(Memory::REGULAR, dtype, var_name,
                                    buffer_name,
                                    codegen_state.index.index_after_trace, true,
                                    codegen_state.index.pred_after_trace);
}

void CUDAEmitter::emit_reduce(const Op* node,
                              const CodegenState& codegen_state) {
  this->code_stream << node->get_implementation()->generate();
}

void CUDAEmitter::emit_reduce_window(const Op* node,
                                     const CodegenState& codegen_state) {
  const ReduceWindow* reduce_window = node->as<ReduceWindow>();
  Dtype dtype = node->get_output_spec(0).get_dtype();

  std::string reduce_window_node_name =
      (reduce_window->get_name() + codegen_state.node_name_suffix);

  this->code_stream << mononn_engine::helpers::string_format(
      "%s %s = %s;\n", dtype.to_string().c_str(),
      reduce_window_node_name.c_str(),
      reduce_window->get_init_value().get_value().c_str());

  std::vector<std::vector<int>> window_positions =
      reduce_window->get_window_positions();

  for (auto const& window_position : window_positions) {
    std::string operand_name =
        node->get_operand(0)->get_name() + codegen_state.node_name_suffix +
        "_window_pos_" +
        mononn_engine::helpers::join<int>(
            "_", window_position,
            [](const int& v) -> std::string { return std::to_string(v); });
    this->code_stream << mononn_engine::helpers::string_format(
        "%s = %s(%s, %s);\n", reduce_window_node_name.c_str(),
        reduce_window->get_reduction_functor_generator()
            ->instance_name()
            .c_str(),
        reduce_window_node_name.c_str(), operand_name.c_str());
  }
}

void CUDAEmitter::emit_select(const Op* node,
                              const CodegenState& codegen_state) {
  std::string functor_name =
      Functor::get_functor_name_for_op_type(node->get_type());
  Functor functor(functor_name, node->get_output_spec(0).get_dtype());
  FunctionInvocation invocation(functor.get_name());
  invocation.add_arg(node->get_operand(0)->get_name() +
                     codegen_state.node_name_suffix);
  invocation.add_arg(node->get_operand(1)->get_name() +
                     codegen_state.node_name_suffix);
  invocation.add_arg(node->get_operand(2)->get_name() +
                     codegen_state.node_name_suffix);

  // this->code_stream << invocation.to_string();
  this->code_stream << mononn_engine::helpers::string_format(
      "%s %s = %s;\n", node->get_output_spec(0).get_dtype().to_string().c_str(),
      (node->get_name() + codegen_state.node_name_suffix).c_str(),
      invocation.to_string().c_str());
}

void CUDAEmitter::emit_non_op(const Op* node,
                              const CodegenState& codegen_state) {
  this->code_stream << mononn_engine::helpers::string_format(
      "%s %s = %s;\n", node->get_output_spec(0).get_dtype().to_string().c_str(),
      (node->get_name() + codegen_state.node_name_suffix).c_str(),
      (node->get_operand(0)->get_name() + codegen_state.node_name_suffix)
          .c_str());
}

void CUDAEmitter::emit_output(const Op* node,
                              const CodegenState& codegen_state) {
  std::string var_name = node->get_name() + codegen_state.node_name_suffix;
  std::string buffer_name =
      BufferManager::get_buffer_name(node->get_name()) + "_output";
  auto dtype = node->get_output_spec(0).get_dtype();
  this->code_stream << Memory::write(Memory::REGULAR, dtype, var_name,
                                     buffer_name,
                                     codegen_state.index.index_before_trace);
}

void CUDAEmitter::emit_output(const Op* node,
                              const ConcreteIndexStamp& concrete_index) {
  // std::vector<CodegenState> symbolic_codegen_state_list;

  // for (auto const index : node->get_symbolic_index()) {
  //     CodegenState codegen_state;
  //     codegen_state.node_name_suffix = "";
  //     codegen_state.index = index.instantiate(symbolic_index_initializer);
  //     symbolic_codegen_state_list.push_back(codegen_state);
  // }

  CodegenState codegen_state;
  codegen_state.node_name_suffix = "";
  codegen_state.index = concrete_index;

  if (node->get_type() == OpType::reduce) {
    this->emit_output(node, codegen_state);
  } else {
    std::vector<CodegenState> concrete_codegen_state_list =
        this->codegens_state_space->generate_codegen_state(codegen_state);

    for (auto const& state : concrete_codegen_state_list) {
      this->emit_output(node, state);
    }
  }
}

void CUDAEmitter::emit_loop_end(const Op* node) { this->code_stream << "}\n"; }

void CUDAEmitter::emit_post_reduce_if(const Op* node) {
  this->code_stream << node->get_implementation()
                           ->as<ReduceImpl>()
                           ->get_post_reduce_if_statement()
                    << "\n";
}

void CUDAEmitter::emit_post_reduce_if_end(const Op* node) {
  this->code_stream
      << node->get_implementation()->as<ReduceImpl>()->get_post_reduce_if_end()
      << "\n";
}

void CUDAEmitter::emit_reduce_broadcast(const Op* node) {
  this->code_stream
      << node->get_implementation()->as<ReduceImpl>()->generate_reduce();
}

const std::stringstream& CUDAEmitter::get_code_stream() const {
  return this->code_stream;
}

CodegenState CodegenStateSampler::sample(
    const CodegenState& codegen_state) const {
  return this->sample_func(codegen_state);
}

void CodegenStateSpace::push(const StateSamplerArray& state_array) {
  this->space_array.push_back(state_array);
}

void CodegenStateSpace::pop() { this->space_array.pop_back(); }

std::vector<CodegenState> CodegenStateSpace::generate_codegen_state(
    const CodegenState& init_state) const {
  if (this->space_array.empty()) {
    return {init_state};
  }

  std::vector<std::vector<int>> space_selection_index(
      this->space_array[0].size());
  // space_selection_index.push_back({});

  for (int idx_space_sampler = 0;
       idx_space_sampler < this->space_array[0].size(); ++idx_space_sampler) {
    space_selection_index[idx_space_sampler].push_back(idx_space_sampler);
  }

  for (int idx_space_array = 1; idx_space_array < this->space_array.size();
       ++idx_space_array) {
    std::vector<int> space_sampiler_index(
        this->space_array[idx_space_array].size());

    std::iota(space_sampiler_index.begin(), space_sampiler_index.end(), 0);

    space_selection_index = mononn_engine::helpers::cartesian_join<
        std::vector<int>, int, std::vector<int>>(
        space_selection_index, space_sampiler_index,
        [](const std::vector<int>& vec, const int& value) -> std::vector<int> {
          std::vector<int> ret = vec;
          ret.push_back(value);

          return ret;
        });
  }

  std::vector<CodegenState> result_state_list;

  for (auto index_list : space_selection_index) {
    CodegenState final_state = init_state;

    for (int idx = 0; idx < index_list.size(); ++idx) {
      final_state = this->space_array[idx][index_list[idx]].sample(final_state);
    }

    result_state_list.push_back(final_state);
  }

  return result_state_list;
}

std::vector<CodegenState> CodegenStateSpace::generate_codegen_state(
    const std::vector<CodegenState>& init_state_list) const {
  std::vector<CodegenState> result_state_list;

  for (auto const& init_state : init_state_list) {
    std::vector<CodegenState> partial_state_list =
        this->generate_codegen_state(init_state);
    result_state_list.insert(result_state_list.end(),
                             partial_state_list.begin(),
                             partial_state_list.end());
  }

  return result_state_list;
}

void CodegenStateSpaceMgr::emit_op(const Op* node) {
  if (node->get_type() == OpType::reduce_window) {
    this->emit_reduce_window(node);
  } else if (node->get_type() == OpType::slice) {
    this->emit_slice(node);
  }
}

CodegenStateSpaceMgr::CodegenStateSpaceMgr() {
  this->codegen_state_space = std::make_unique<CodegenStateSpace>();
}

void CodegenStateSpaceMgr::emit_instruction_level_parallelism(int ilp_factor) {
  CodegenStateSpace::StateSamplerArray state_sampler_array;

  for (int ilp_id = 0; ilp_id < ilp_factor; ++ilp_id) {
    state_sampler_array.emplace_back(
        [ilp_id](const CodegenState& codegen_state) -> CodegenState {
          CodegenState next_codegen_state;

          next_codegen_state.node_name_suffix =
              codegen_state.node_name_suffix + "__i" + std::to_string(ilp_id);
          next_codegen_state.index = codegen_state.index.instantiate(
              {{"ilp_index_id", std::to_string(ilp_id)}});

          return next_codegen_state;
        });
  }

  this->codegen_state_space->push(state_sampler_array);
}

void CodegenStateSpaceMgr::emit_reduce_window(const Op* node) {
  const ReduceWindow* reduce_window = node->as<ReduceWindow>();

  CodegenStateSpace::StateSamplerArray state_sampler_array;

  std::vector<std::vector<int>> window_positions =
      reduce_window->get_window_positions();

  for (auto const& window_position : window_positions) {
    state_sampler_array.emplace_back(
        [window_position](const CodegenState& codegen_state) -> CodegenState {
          CodegenState next_codegen_state;

          std::map<std::string, std::string> params;

          for (int idx = 0; idx < window_position.size(); ++idx) {
            std::string key = "window_position_" + std::to_string(idx);
            params[key] = std::to_string(window_position[idx]);
          }

          next_codegen_state.node_name_suffix =
              codegen_state.node_name_suffix + "_window_pos_" +
              mononn_engine::helpers::join<int>(
                  "_", window_position, [](const int& v) -> std::string {
                    return std::to_string(v);
                  });
          next_codegen_state.index = codegen_state.index.instantiate(params);

          return next_codegen_state;
        });
  }

  this->codegen_state_space->push(state_sampler_array);
}

void CodegenStateSpaceMgr::emit_slice(const Op* node) {
  LOG(FATAL) << "CodegenStateSpaceMgr::emit_slice unimplemented";
}

void CodegenStateSpaceMgr::recall_op(const Op* node) {
  if (node->get_type() == OpType::reduce_window) {
    this->recall_reduce_window(node);
  } else if (node->get_type() == OpType::slice) {
    this->recall_slice(node);
  }
}

void CodegenStateSpaceMgr::recall_reduce_window(const Op* node) {
  this->codegen_state_space->pop();
}

void CodegenStateSpaceMgr::recall_slice(const Op* node) {
  LOG(FATAL) << "CodegenStateSpaceMgr::recall_slice unimplemented";
}

const CodegenStateSpace* CodegenStateSpaceMgr::get_codegen_state_space() const {
  return this->codegen_state_space.get();
}
}  // namespace codegen
}  // namespace mononn_engine