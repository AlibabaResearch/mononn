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

#include "mononn_engine/codegen/graph_codegen.h"

#include <sstream>

#include "mononn_engine/codegen/cluster_codegen.h"
#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/codegen/host_codegen.h"
#include "mononn_engine/codegen/model_data.h"
#include "mononn_engine/codegen/node_codegen.h"
#include "mononn_engine/codegen/reduction_functor_generator.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/gpu/buffer_manager.h"
#include "mononn_engine/core/gpu/defined.h"
#include "mononn_engine/core/gpu/functor.h"
#include "mononn_engine/core/gpu/headers.h"
#include "mononn_engine/core/gpu/limits.h"
#include "mononn_engine/core/gpu/multi_buffer.h"
#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/constant.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/core/op_impl/reduce_impl.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/helpers/string_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace mononn_engine {
namespace codegen {
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using BufferManager = mononn_engine::core::gpu::BufferManager;
using Op = mononn_engine::core::op::Op;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using Edge = mononn_engine::core::edge::Edge<Op>;
using Synchronization = mononn_engine::core::gpu::Synchronization;
using OpType = mononn_engine::core::op::OpType;
using CUDADefined = mononn_engine::core::gpu::CUDADefined;
using Headers = mononn_engine::core::gpu::Headers;
using Functor = mononn_engine::core::gpu::Functor;
using ReduceImpl = mononn_engine::core::op_impl::ReduceImpl;
using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
using HostCodegen = mononn_engine::codegen::HostCodegen;
using Limits = mononn_engine::core::gpu::Limits;
using Constant = mononn_engine::core::op::Constant;
using Synchronization = mononn_engine::core::gpu::Synchronization;
using Config = mononn_engine::config::Config;
using MultiBuffer = mononn_engine::core::gpu::MultiBuffer;
using Dtype = mononn_engine::core::tensor::Dtype;
using NodeCodegen = mononn_engine::codegen::NodeCodegen;
using ReductionFunctorGenerator =
    mononn_engine::codegen::ReductionFunctorGenerator;
using EnvVar = mononn_engine::helpers::EnvVar;

std::string get_xla_constant_buffer_name(std::string node_name) {
  return "buffer_for_" + node_name;
}

std::string get_constant_buffer_initialization_for_tf_xla_buffer(
    const Graph* graph) {
  std::stringstream ss;

  for (auto const& node_name : graph->get_node_list()) {
    auto node = graph->get_node(node_name);
    if (node->get_type() == OpType::constant) {
      // if (node->as<Constant>()->is_scalar()) { continue; }
      ss << "__device__ uint8_t " << get_xla_constant_buffer_name(node_name)
         << " [";
      ss << node->get_output_buffer_size_in_bytes() << "];\n";
    }
  }

  return ss.str();
}

std::string initialize_buffer_pointers(const GraphCodegen::Params& params) {
  std::stringstream ss;

  ss << "// initialize_buffer_pointers\n";

  if (params.buffer_management_policy ==
      GraphCodegen::MONONN_BUFFER_MANAGEMENT_DEFAULT) {
    std::vector<std::string> buffer_node_list;
    MultiBuffer onefuser_buffer(Config::get()->onefuser_buffer_name);
    for (auto const& node_name :
         BufferManager::get_buffered_nodes_in_global()) {
      if (!params.graph->has_node(node_name)) continue;
      std::shared_ptr<Op> node = params.graph->get_node(node_name);
      buffer_node_list.push_back(node_name);
      onefuser_buffer.add_buffer(node->get_output_buffer_size_in_bytes());
    }

    for (int idx = 0; idx < buffer_node_list.size(); ++idx) {
      std::string node_name = buffer_node_list[idx];
      std::string buffer_name = BufferManager::get_buffer_name(node_name);
      ss << mononn_engine::helpers::string_format(
                "void *%s = %s;", buffer_name.c_str(),
                onefuser_buffer.get_pointer_to_buffer(idx).c_str())
         << "\n";
    }
  } else if (params.buffer_management_policy ==
             GraphCodegen::MONONN_BUFFER_MANAGEMENT_TF_XLA) {
    using AllocationInfo = std::pair<const xla::BufferAllocation*,
                                     xla::BufferAllocation::OffsetSize>;
    std::unordered_map<std::string, AllocationInfo>
        original_node_name_to_allocation_info;
    // this is for elements inside tuplle.
    // for example: fusion.1 = (f32[100, 100], f32[50, 50])
    // fusion.1{} goes to original_node_name_to_allocation_info
    // fusion.1{0} goes to original_node_name_to_allocation_info_tuple_elements
    // fusion.1{1} goes to original_node_name_to_allocation_info_tuple_elements
    std::unordered_map<std::string, std::unordered_map<int64_t, AllocationInfo>>
        original_node_name_to_allocation_info_tuple_elements;

    for (auto const& allocation : *params.allocation_list) {
      for (auto const& [hlo_value, offset] : allocation.assigned_buffers()) {
        if (hlo_value->defining_instruction()->shape().IsTuple()) {
          if (hlo_value->index().empty()) {
            std::string node_name =
                mononn_engine::helpers::get_canonicalized_node_name(
                    hlo_value->defining_instruction()->name());
            original_node_name_to_allocation_info[node_name] =
                std::make_pair(&allocation, offset);
          } else {
            if (hlo_value->index().size() != 1) {
              LOG(FATAL)
                  << "hlo value: " << hlo_value->ToShortString()
                  << " have too many index whereas only one index is expected";
            }

            std::string node_name =
                mononn_engine::helpers::get_canonicalized_node_name(
                    hlo_value->defining_instruction()->name());
            int64_t index_position = hlo_value->index()[0];
            if (!original_node_name_to_allocation_info_tuple_elements.count(
                    node_name)) {
              original_node_name_to_allocation_info_tuple_elements[node_name] =
                  std::unordered_map<
                      int64_t, std::pair<const xla::BufferAllocation*,
                                         xla::BufferAllocation::OffsetSize>>();
            }

            original_node_name_to_allocation_info_tuple_elements
                [node_name][index_position] =
                    std::make_pair(&allocation, offset);
          }
        } else {
          std::string node_name =
              mononn_engine::helpers::get_canonicalized_node_name(
                  hlo_value->defining_instruction()->name());
          original_node_name_to_allocation_info[node_name] =
              std::make_pair(&allocation, offset);
        }
      }
    }

    // Respect XLA origional alias analysis.
    for (auto const& inst :
         params.hlo_module->entry_computation()->instructions()) {
      if (inst->opcode() == xla::HloOpcode::kBitcast ||
          inst->opcode() == xla::HloOpcode::kGetTupleElement) {
        std::string node_name =
            mononn_engine::helpers::get_canonicalized_node_name(inst->name());

        auto buffers = params.alias_analysis->GetUniqueBufferAt(inst).values();

        if (buffers.size() != 1) {
          LOG(FATAL) << "Node " << inst->name()
                     << " get multiple aliased values";
        }

        auto const aliased_inst = buffers[0]->defining_instruction();
        std::string aliased_node_name =
            mononn_engine::helpers::get_canonicalized_node_name(
                aliased_inst->name());

        if (inst->opcode() == xla::HloOpcode::kBitcast) {
          if (aliased_inst->shape().IsTuple()) {
            // Tuple -> GTE -> bitcast pattern
            if (inst->operand(0)->opcode() ==
                xla::HloOpcode::kGetTupleElement) {
              if (!original_node_name_to_allocation_info_tuple_elements.count(
                      aliased_node_name)) {
                LOG(FATAL) << "original_node_name_to_allocation_info_tuple_"
                              "elements do not have "
                           << aliased_node_name;
              }

              if (!original_node_name_to_allocation_info_tuple_elements
                       [aliased_node_name]
                           .count(inst->operand(0)->tuple_index())) {
                LOG(FATAL)
                    << "original_node_name_to_allocation_info_tuple_elements["
                    << aliased_node_name << "]"
                    << " do not have " << inst->operand(0)->tuple_index();
              }

              original_node_name_to_allocation_info[node_name] =
                  original_node_name_to_allocation_info_tuple_elements
                      [aliased_node_name][inst->operand(0)->tuple_index()];
            } else {
              LOG(FATAL) << "Bitcast instruction " << inst->name()
                         << " aliased to a tuple output "
                         << aliased_inst->name();
            }
          }

          if (!original_node_name_to_allocation_info.count(aliased_node_name)) {
            LOG(FATAL) << "original_node_name_to_allocation_info do not have "
                       << aliased_node_name;
          }

          original_node_name_to_allocation_info[node_name] =
              original_node_name_to_allocation_info[aliased_node_name];
        } else if (inst->opcode() == xla::HloOpcode::kGetTupleElement) {
          if (!aliased_inst->shape().IsTuple()) {
            LOG(FATAL) << "GetTupleElement instruction " << inst->name()
                       << "aliased to a nono-tuple output "
                       << aliased_inst->name();
          }

          if (!original_node_name_to_allocation_info_tuple_elements.count(
                  aliased_node_name)) {
            LOG(FATAL) << "original_node_name_to_allocation_info_tuple_"
                          "elements do not have "
                       << aliased_node_name;
          }

          if (!original_node_name_to_allocation_info_tuple_elements
                   [aliased_node_name]
                       .count(inst->tuple_index())) {
            LOG(FATAL)
                << "original_node_name_to_allocation_info_tuple_elements["
                << aliased_node_name << "]"
                << " do not have " << inst->tuple_index();
          }

          original_node_name_to_allocation_info[node_name] =
              original_node_name_to_allocation_info_tuple_elements
                  [aliased_node_name][inst->tuple_index()];
        } else {
          LOG(FATAL) << "Opcode: " << inst->opcode() << " unexpected.";
        }
      }
    }

    // // This code section should respect all alias analysis.
    // for (auto const &inst :
    // params.hlo_module->entry_computation()->instructions()) {
    //     if (inst->opcode() == xla::HloOpcode::kBitcast) {
    //         std::string node_name
    //                     =
    //                     mononn_engine::helpers::get_canonicalized_node_name(inst->name());

    //         auto operand_inst = inst->operand(0);
    //         // operand_node -> get tuple element -> bitcast node
    //         if (operand_inst->opcode() == xla::HloOpcode::kGetTupleElement) {
    //             std::string operand_node_name
    //                     =
    //                     mononn_engine::helpers::get_canonicalized_node_name(operand_inst->operand(0)->name());
    //             if
    //             (!original_node_name_to_allocation_info_tuple_elements.count(operand_node_name))
    //             {
    //                 LOG(FATAL) << "Operand node of " << node_name << "/" <<
    //                 operand_inst->name() << " not in
    //                 original_node_name_to_allocation_info_tuple_elements,
    //                 operand node name "
    //                             << operand_node_name;
    //             }

    //             if
    //             (!original_node_name_to_allocation_info_tuple_elements[operand_node_name].count(operand_inst->tuple_index()))
    //             {
    //                 LOG(FATAL) <<
    //                 "original_node_name_to_allocation_info_tuple_elements["
    //                         << operand_node_name << "]" << " do not have
    //                         tuple index " << operand_inst->tuple_index();
    //             }

    //             original_node_name_to_allocation_info[node_name] =
    //             original_node_name_to_allocation_info_tuple_elements[operand_node_name][operand_inst->tuple_index()];
    //         } else { // operand_node -> bitcast node
    //             std::string operand_node_name
    //                     =
    //                     mononn_engine::helpers::get_canonicalized_node_name(operand_inst->name());
    //             if
    //             (!original_node_name_to_allocation_info.count(operand_node_name))
    //             {
    //                 LOG(FATAL) << "Operand node of " << node_name << " not in
    //                 original_node_name_to_allocation_info, operand node name
    //                 " << operand_node_name;
    //             }

    //             original_node_name_to_allocation_info[node_name] =
    //             original_node_name_to_allocation_info[operand_node_name];
    //         }
    //     }
    // }

    // map instruction name for instrutions inside fusion to it's parent
    // instruciton;
    std::unordered_map<std::string, xla::HloInstruction*>
        hlo_inst_name_to_buffer_defining_instruction;
    {
      for (auto const& inst :
           params.hlo_module->entry_computation()->instructions()) {
        if (inst->opcode() == xla::HloOpcode::kFusion) {
          for (auto const& inner_inst :
               inst->fused_instructions_computation()->instructions()) {
            hlo_inst_name_to_buffer_defining_instruction[inner_inst->name()] =
                inst;
          }
        } else {
          hlo_inst_name_to_buffer_defining_instruction[inst->name()] =
              inst;  // Top level instruction define its own buffer.
        }
      }
    }

    auto find_allocation_info_for_output_node_in_cluster_node =
        [&hlo_inst_name_to_buffer_defining_instruction, &params,
         &original_node_name_to_allocation_info,
         &original_node_name_to_allocation_info_tuple_elements](
            const std::string& output_node_name,
            const std::string& output_node_hlo_inst_name,
            absl::optional<int> tuple_index_for_non_type_type_output_node =
                absl::nullopt) -> AllocationInfo {
      if (!hlo_inst_name_to_buffer_defining_instruction.count(
              output_node_hlo_inst_name)) {
        LOG(FATAL) << output_node_hlo_inst_name
                   << " not found in any fused computation "
                   << params.hlo_module->name();
      }

      auto fused_inst = hlo_inst_name_to_buffer_defining_instruction
          [output_node_hlo_inst_name];

      if (fused_inst->shape().IsTuple()) {
        auto root_inst =
            fused_inst->opcode() == xla::HloOpcode::kFusion
                ? fused_inst->fused_instructions_computation()
                      ->root_instruction()
                : fused_inst;  // fused_inst may not be fusion inst. It may be
                               // clustered by ClusterSingleNodePass.

        if (root_inst->opcode() != xla::HloOpcode::kTuple) {
          if (!tuple_index_for_non_type_type_output_node.has_value()) {
            LOG(FATAL) << "Tuple index must be specified explicit for "
                          "non-tuple node with tuple shaped output";
          }

          std::string fuse_inst_node_name =
              mononn_engine::helpers::get_canonicalized_node_name(
                  fused_inst->name());

          if (!original_node_name_to_allocation_info_tuple_elements.count(
                  fuse_inst_node_name)) {
            LOG(FATAL) << "original_node_name_to_allocation_info_tuple_"
                          "elements do not have "
                       << fuse_inst_node_name;
          }

          if (!original_node_name_to_allocation_info_tuple_elements
                   [fuse_inst_node_name]
                       .count(
                           tuple_index_for_non_type_type_output_node.value())) {
            LOG(FATAL)
                << "original_node_name_to_allocation_info_tuple_elements["
                << fuse_inst_node_name << "] do not have GTE id "
                << tuple_index_for_non_type_type_output_node.value();
          }

          return original_node_name_to_allocation_info_tuple_elements
              [fuse_inst_node_name]
              [tuple_index_for_non_type_type_output_node.value()];
        }

        auto output_node_it = std::find_if(
            root_inst->operands().begin(), root_inst->operands().end(),
            [&output_node_hlo_inst_name](
                xla::HloInstruction* const operand_inst) -> bool {
              return operand_inst->name() == output_node_hlo_inst_name;
            });

        if (output_node_it == root_inst->operands().end()) {
          LOG(FATAL) << output_node_hlo_inst_name
                     << " not found in the root instruction "
                     << root_inst->name();
        }

        int64_t get_tuple_element_id =
            output_node_it - root_inst->operands().begin();
        std::string fuse_inst_node_name =
            mononn_engine::helpers::get_canonicalized_node_name(
                fused_inst->name());

        if (!original_node_name_to_allocation_info_tuple_elements.count(
                fuse_inst_node_name)) {
          LOG(FATAL) << "original_node_name_to_allocation_info_tuple_elements "
                        "do not have "
                     << fuse_inst_node_name;
        }

        if (!original_node_name_to_allocation_info_tuple_elements
                 [fuse_inst_node_name]
                     .count(get_tuple_element_id)) {
          LOG(FATAL) << "original_node_name_to_allocation_info_tuple_elements["
                     << fuse_inst_node_name << "] do not have GTE id "
                     << get_tuple_element_id;
        }

        return original_node_name_to_allocation_info_tuple_elements
            [fuse_inst_node_name][get_tuple_element_id];
      } else {
        std::string fuse_inst_node_name =
            mononn_engine::helpers::get_canonicalized_node_name(
                fused_inst->name());
        if (!original_node_name_to_allocation_info.count(fuse_inst_node_name)) {
          LOG(FATAL) << fuse_inst_node_name
                     << " not found in original_node_name_to_allocation_info";
        }

        return original_node_name_to_allocation_info[fuse_inst_node_name];
      }
    };

    // initialize buffer pointers
    for (auto const& node_name : params.graph->get_node_list()) {
      auto node = params.graph->get_node(node_name);
      if (node->get_type() == OpType::parameter ||
          node->get_type() == OpType::get_tuple_element ||
          node->get_type() == OpType::global_sync) {
        continue;
      }

      if (node->get_type() == OpType::constant) {
        if (node->as<Constant>()->is_scalar()) {
          continue;
        }

        std::string buffer_name = get_xla_constant_buffer_name(node_name);
        ss << mononn_engine::helpers::string_format(
            "void *%s = reinterpret_cast<void *>((&reinterpret_cast<uint8_t "
            "*>(%s)[%lld]));\n",
            node_name.c_str(), buffer_name.c_str(), int64_t(0));
        continue;
      }

      if (node->get_output_specs_count() == 1) {
        if (node->get_type() == OpType::cluster) {
          std::string output_node_name =
              node->as<ClusterOp>()->get_graph_ptr()->get_output_node(0);
          auto output_node = node->as<ClusterOp>()->get_graph_ptr()->get_node(
              output_node_name);
          auto allocation_info =
              find_allocation_info_for_output_node_in_cluster_node(
                  output_node_name, output_node->get_hlo_instruction_name());
          std::string buffer_name = params.allocation_ptr_to_buffer_name.at(
              reinterpret_cast<uint64_t>(allocation_info.first));

          ss << mononn_engine::helpers::string_format(
              "void *%s = reinterpret_cast<void *>((&reinterpret_cast<uint8_t "
              "*>(%s)[%lld]));\n",
              node_name.c_str(), buffer_name.c_str(),
              allocation_info.second.offset);
        } else if (node->get_type() == OpType::custom_call) {
          auto allocation_info =
              original_node_name_to_allocation_info[node_name];
          std::string buffer_name = params.allocation_ptr_to_buffer_name.at(
              reinterpret_cast<uint64_t>(allocation_info.first));
          ss << mononn_engine::helpers::string_format(
              "void *%s = reinterpret_cast<void *>((&reinterpret_cast<uint8_t "
              "*>(%s)[%lld]));\n",
              node_name.c_str(), buffer_name.c_str(),
              allocation_info.second.offset);
        } else {
          LOG(FATAL) << "Unexpected node type:" << node->get_type().to_string();
        }
      } else if (node->get_output_specs_count() > 1) {
        if (node->get_type() == OpType::cluster) {
          ss << "// Get tuple element nodes for " << node_name << "\n";

          std::vector<std::string> output_node_name_list =
              node->as<ClusterOp>()->get_graph_ptr()->get_output_nodes();

          // Node have one graph output node but have multiple output tensor
          // spec. This indicate the output node of the graph is a non-tuple
          // node with tuple output shape Such as reduction with multiple
          // operands.
          if (output_node_name_list.size() == 1) {
            for (int idx = 0; idx < node->get_output_specs_count(); ++idx) {
              auto output_node =
                  node->as<ClusterOp>()->get_graph_ptr()->get_node(
                      output_node_name_list[0]);
              auto allocation_info =
                  find_allocation_info_for_output_node_in_cluster_node(
                      output_node_name_list[0],
                      output_node->get_hlo_instruction_name(), idx);
              std::string GTE_node_name = mononn_engine::helpers::string_format(
                  "get_tuple_element_%s_%d", node_name.c_str(), idx);
              std::string buffer_name = params.allocation_ptr_to_buffer_name.at(
                  reinterpret_cast<uint64_t>(allocation_info.first));

              ss << mononn_engine::helpers::string_format(
                  "void *%s = reinterpret_cast<void "
                  "*>((&reinterpret_cast<uint8_t *>(%s)[%lld]));\n",
                  GTE_node_name.c_str(), buffer_name.c_str(),
                  allocation_info.second.offset);
            }
          } else {
            for (int idx = 0; idx < output_node_name_list.size(); ++idx) {
              auto output_node =
                  node->as<ClusterOp>()->get_graph_ptr()->get_node(
                      output_node_name_list[idx]);
              auto allocation_info =
                  find_allocation_info_for_output_node_in_cluster_node(
                      output_node_name_list[idx],
                      output_node->get_hlo_instruction_name());

              std::string GTE_node_name = mononn_engine::helpers::string_format(
                  "get_tuple_element_%s_%d", node_name.c_str(), idx);
              std::string buffer_name = params.allocation_ptr_to_buffer_name.at(
                  reinterpret_cast<uint64_t>(allocation_info.first));

              ss << mononn_engine::helpers::string_format(
                  "void *%s = reinterpret_cast<void "
                  "*>((&reinterpret_cast<uint8_t *>(%s)[%lld]));\n",
                  GTE_node_name.c_str(), buffer_name.c_str(),
                  allocation_info.second.offset);
            }
          }

        } else if (node->get_type() == OpType::custom_call) {
          // This case should be covered when initialize original GET nodes (see
          // below).

          // auto custom_call_inst =
          // params.hlo_module->entry_computation()->GetInstructionWithName(node->get_hlo_instruction_name());
          // auto users = custom_call_inst->users();

          // if (std::any_of(users.begin(), users.end(), [](const
          // xla::HloInstruction *inst) -> bool {
          //     return inst->opcode() != xla::HloOpcode::kGetTupleElement;
          // })) {
          //     LOG(FATAL) << "Multi output inst " << node_name << " have node
          //     other than get tuple element";
          // }

          // if
          // (!original_node_name_to_allocation_info_tuple_elements.count(node_name))
          // {
          //     LOG(FATAL) << "Node " << node_name << " not found in
          //     original_node_name_to_allocation_info_tuple_elements";
          // }

          // ss << "// Get tuple element nodes for " << node_name << "\n";

          // for (auto const &inst : users) {
          //     if
          //     (!original_node_name_to_allocation_info_tuple_elements[node_name].count(inst->parameter_number()))
          //     {
          //         LOG(FATAL) << "Node " << node_name << " have no parameter
          //         number " << inst->parameter_number();
          //     }

          //     std::string get_tuple_element_node_name =
          //     mononn_engine::helpers::get_canonicalized_node_name(inst->name());

          //     auto allocation_info =
          //     original_node_name_to_allocation_info_tuple_elements[node_name][inst->parameter_number()];
          //     std::string buffer_name =
          //     params.allocation_ptr_to_buffer_name.at(reinterpret_cast<uint64_t>(allocation_info.first));
          //     ss << mononn_engine::helpers::string_format("void *%s =
          //     reinterpret_cast<void *>((&reinterpret_cast<uint8_t
          //     *>(%s)[%lld]));\n",
          //         get_tuple_element_node_name.c_str(), buffer_name.c_str(),
          //         allocation_info.second.offset);
          // }
        } else {
          LOG(FATAL) << "Unexpected node type:" << node->get_type().to_string();
        }
      } else {
        LOG(FATAL) << "Unexpected output count: "
                   << node->get_output_specs_count();
      }
    }

    // Initialize buffer pointers for original GET nodes.
    ss << "// Initialize buffer pointers for original GET nodes\n";
    for (auto const& inst :
         params.hlo_module->entry_computation()->instructions()) {
      if (inst->opcode() == xla::HloOpcode::kGetTupleElement) {
        std::string get_tuple_element_node_name =
            mononn_engine::helpers::get_canonicalized_node_name(inst->name());
        auto allocation_info =
            original_node_name_to_allocation_info[get_tuple_element_node_name];
        std::string buffer_name = params.allocation_ptr_to_buffer_name.at(
            reinterpret_cast<uint64_t>(allocation_info.first));
        ss << mononn_engine::helpers::string_format(
            "void *%s = reinterpret_cast<void *>((&reinterpret_cast<uint8_t "
            "*>(%s)[%lld]));\n",
            get_tuple_element_node_name.c_str(), buffer_name.c_str(),
            allocation_info.second.offset);
      }
    }
  } else {
    LOG(FATAL) << "Unrecognized policy " << params.buffer_management_policy;
  }

  return ss.str();
}

std::string generate_reduction_functors() {
  std::stringstream ss;

  for (auto const& [computation_name, generator] :
       ReductionFunctorGenerator::Registry()->get_generator()) {
    ss << "//=====Reduction functor used by " << computation_name << "=====\n";
    ss << generator->generate_functor_definition() << "\n";
  }

  return ss.str();
}

std::unique_ptr<CUDAProgram> GraphCodegen::generate(const Params& params) {
  LOG(INFO) << "Begin code generation...";
  bool TF_MONONN_ENABLED = EnvVar::is_true("TF_MONONN_ENABLED");

  std::unique_ptr<CUDAProgram> cuda_program =
      std::make_unique<CUDAProgram>(params.cuda_context);

  params.graph->build_transitive_closure();

  int block_size = params.cuda_context->cuda_runtime_context.block_dim.XYZ();
  int block_count = params.cuda_context->cuda_runtime_context.grid_dim.XYZ();

  int sm_count = params.cuda_context->cuda_device_context.sm_count;

  cuda_program->file_ref("headers.cuh") << "#pragma once"
                                        << "\n";
  cuda_program->file_ref("headers.cuh") << Headers::get_headers() << "\n";
  cuda_program->file_ref("headers.cuh")
      << Headers::get_headers_main_only() << "\n";
  cuda_program->file_ref("headers.cuh") << Headers::get_cuda_helpers() << "\n";
  cuda_program->file_ref("headers.cuh") << Headers::get_tuple_headers() << "\n";
  cuda_program->file_ref("headers.cuh")
      << Headers::get_tuple_shfl_headers() << "\n";
  cuda_program->file_ref("headers.cuh")
      << params.cuda_context->cuda_device_context.get_cuda_arch_global_macro()
      << "\n";
  cuda_program->file_ref("headers.cuh")
      << ReduceImpl::get_prerequisite_definition() << "\n\n";
  cuda_program->file_ref("headers.cuh")
      << GemmImpl::get_prerequisite_definition() << "\n\n";
  cuda_program->file_ref("headers.cuh")
      << Synchronization::get_prerequisite_definition() << "\n\n";
  cuda_program->file_ref("headers.cuh")
      << Functor::get_all_functors_definition() << "\n\n";

  if (mononn_engine::core::gpu::cutlass::Arch::newer_or_equal(
          params.cuda_context->cuda_device_context.get_cutlass_arch_tag(),
          mononn_engine::core::gpu::cutlass::Arch::Sm80)) {
    cuda_program->file_ref("headers.cuh") << Headers::get_async_copy_headers();
  }

  cuda_program->file_ref("headers.cuh")
      << generate_reduction_functors() << "\n\n";

  cuda_program->file_ref("main.cu") << "#include \"headers.cuh\""
                                    << "\n";

  params.graph->wave_front_order(
      [&](std::shared_ptr<Op> node, std::shared_ptr<Op> next_node) -> void {
        if (node->get_type() == OpType::cluster) {
          if (params.codegen_reject_list.find(node->get_name()) !=
              params.codegen_reject_list.end()) {
            return;
          }

          ClusterCodegen::setup_codegen(
              params.cuda_context, std::static_pointer_cast<ClusterOp>(node));
        }
      });

  params.graph->wave_front_order(
      [&](std::shared_ptr<const Op> node,
          std::shared_ptr<const Op> next_node) -> void {
        if (node->get_type() == OpType::cluster) {
          if (params.codegen_reject_list.find(node->get_name()) !=
              params.codegen_reject_list.end()) {
            return;
          }

          if (TF_MONONN_ENABLED) {
            cuda_program->file_ref("main.cu")
                << ClusterCodegen::generate_function_definition(
                       params.cuda_context,
                       std::static_pointer_cast<const ClusterOp>(node))
                << "\n\n";
          } else {
            std::string cluster_file = node->get_name() + "_computation.cuh";
            cuda_program->file_ref(cluster_file) << "#pragma once"
                                                 << "\n";
            cuda_program->file_ref(cluster_file) << "#include \"headers.cuh\""
                                                 << "\n";
            cuda_program->file_ref(cluster_file)
                << ClusterCodegen::generate_function_definition(
                       params.cuda_context,
                       std::static_pointer_cast<const ClusterOp>(node));
            cuda_program->file_ref("main.cu")
                << "#include \"" << cluster_file << "\""
                << "\n";
          }
        }
      });

  if (params.buffer_management_policy == MONONN_BUFFER_MANAGEMENT_TF_XLA) {
    cuda_program->file_ref("main.cu")
        << get_constant_buffer_initialization_for_tf_xla_buffer(params.graph);
  }

  cuda_program->file_ref("main.cu") << "extern \"C\"\n";
  cuda_program->file_ref("main.cu")
      << mononn_engine::helpers::string_format(
             "__launch_bounds__(%d, %d)", block_size,
             (block_count + sm_count - 1) / sm_count)
      << "\n";
  cuda_program->file_ref("main.cu") << "__global__"
                                    << "\n";
  // cuda_program->file_ref("main.cu") <<
  // mononn_engine::helpers::string_format("void onefuser_kernel(void *%s) {",
  // Config::get()->onefuser_buffer_name.c_str()) << "\n";
  cuda_program->file_ref("main.cu") << mononn_engine::helpers::string_format(
      "void %s(\n", params.kernel_name.c_str());

  for (int idx = 0; idx < params.argument_list.size(); ++idx) {
    if (idx != 0) {
      cuda_program->file_ref("main.cu") << ",\n";
    }

    cuda_program->file_ref("main.cu") << "void *" << params.argument_list[idx];
  }

  cuda_program->file_ref("main.cu") << " ) {\n";

  cuda_program->file_ref("main.cu") << initialize_buffer_pointers(params);

  cuda_program->file_ref("main.cu") << "// Begin kernel\n\n";
  cuda_program->file_ref("main.cu") << "// CUDA defined\n";
  cuda_program->file_ref("main.cu")
      << CUDADefined::initialize(params.cuda_context.get()) << "\n\n";
  cuda_program->file_ref("main.cu") << "// Functor definition\n";
  // cuda_program->file_ref("main.cu") << Functor::get_all_functors_definition()
  // << "\n\n";

  params.graph->wave_front_order(
      [&](std::shared_ptr<const Op> node,
          std::shared_ptr<const Op> next_node) -> void {
        std::string node_name = node->get_name();

        if (params.graph->is_dead_node(node_name)) {
          LOG(WARNING) << "Node " << node_name << "unreachable.";
        }

        if (params.codegen_reject_list.find(node->get_name()) !=
                params.codegen_reject_list.end() &&
            node->get_type() != OpType::get_tuple_element)
          return;

        for (auto const& edge : params.graph->get_node_input_edges(node_name)) {
          if (edge->need_sync()) {
            cuda_program->file_ref("main.cu")
                << edge->get_sync().to_string() << ";\n";
            LOG(WARNING) << "There is un-eliminated synchronization on edge.";
          }
        }

        if (node->get_type() == OpType::cluster) {
          cuda_program->file_ref("main.cu")
              << ClusterCodegen::generate(
                     params.cuda_context,
                     std::static_pointer_cast<const ClusterOp>(node))
              << "\n";
        } else {
          cuda_program->file_ref("main.cu")
              << NodeCodegen::generate(params.cuda_context, node) << "\n";
        }
      });

  cuda_program->file_ref("main.cu") << "//End kernel\n";
  cuda_program->file_ref("main.cu") << "}\n";

  if (params.generate_host_code) {
    cuda_program->file_ref("main.cu") << HostCodegen::generate(
        params.cuda_context, params.graph, params.kernel_name);
  }

  LOG(INFO) << "Code generation completed!";

  if (params.add_model_data) {
    LOG(INFO) << "Adding model data...";

    for (auto const& node_name :
         params.graph->get_node_list_by_type(OpType::constant)) {
      std::shared_ptr<Op> node = params.graph->get_node(node_name);

      if (node->as<Constant>()->is_scalar()) continue;

      auto type = node->get_output_spec(0).get_dtype();
      auto shape = node->get_output_spec(0).get_shape().get_shape();

      std::unique_ptr<ModelData> model_data =
          std::make_unique<ModelData>(node_name + ".npy", type, shape);

      if (type == Dtype::FLOAT32) {
        model_data->add_float_data(node->as<Constant>()->get_data_float());
      } else if (type == Dtype::FLOAT16) {
        model_data->add_half_data(node->as<Constant>()->get_data_half());
      } else {
        LOG(FATAL) << "Unsupported type: " << type.to_string();
      }

      cuda_program->add_model_data(std::move(model_data));
    }

    LOG(INFO) << "Add data completed!";
  }

  return std::move(cuda_program);
}

}  // namespace codegen
}  // namespace mononn_engine