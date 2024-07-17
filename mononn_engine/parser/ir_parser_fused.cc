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

#include "mononn_engine/parser/ir_parser_fused.h"

#include "absl/strings/string_view.h"
#include "mononn_engine/codegen/reduction_functor_generator.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/core/graph/cluster_util.h"
#include "mononn_engine/core/op/all_cluster_operators.h"
#include "mononn_engine/core/op/all_operators.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/reduce_impl.h"
#include "mononn_engine/core/op_impl/reducer.h"
#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/core/tensor/math_op.h"
#include "mononn_engine/helpers/helpers.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/platform/statusor.h"

namespace mononn_engine {
namespace parser {
using Graph = mononn_engine::core::graph::Graph;
using Config = mononn_engine::config::Config;
using Op = mononn_engine::core::op::Op;
using Edge = mononn_engine::core::edge::Edge<Op>;
using TensorShape = mononn_engine::core::tensor::TensorShape;
using MemoryLayout = mononn_engine::core::tensor::MemoryLayout;
using TensorSpec = mononn_engine::core::tensor::TensorSpec;
using Dtype = mononn_engine::core::tensor::Dtype;
using Reducer = mononn_engine::core::op_impl::Reducer;
using MathOp = mononn_engine::core::tensor::MathOp;
using Scalar = mononn_engine::core::tensor::Scalar;
using ClusterElewise = mononn_engine::core::op::ClusterElewise;
using ClusterReduce = mononn_engine::core::op::ClusterReduce;
using Synchronization = mononn_engine::core::gpu::Synchronization;
using ReduceImpl = mononn_engine::core::op_impl::ReduceImpl;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using OpType = mononn_engine::core::op::OpType;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using ClusterUtil = mononn_engine::core::graph::ClusterUtil;
using ReductionFunctorGenerator =
    mononn_engine::codegen::ReductionFunctorGenerator;

#define JOIN_USING(x, y) x::y

#define USING_OP(op_name, op_code, op_class_code, ...) \
  using op_class_code = JOIN_USING(mononn_engine::core::op, op_class_code);

OP_TYPE_LIST(USING_OP)
OP_TYPE_LIST_CLUSTER(USING_OP)

#undef USING_OP
#undef JOIN_USING

std::string hlo_primitive_type_to_string(xla::PrimitiveType type) {
  switch (type) {
    case xla::PRED:
      return "bool";
    case xla::S8:
      return "int8";
    case xla::S16:
      return "int16";
    case xla::S32:
      return "int32";
    case xla::S64:
      return "int64";
    case xla::U8:
      return "uint8";
    case xla::U16:
      return "uint16";
    case xla::U32:
      return "uint32";
    case xla::U64:
      return "uint64";
    case xla::F16:
      return "float16";
    case xla::F32:
      return "float32";
    case xla::F64:
      return "float64";
    default:
      LOG(FATAL) << "Unsupported hlo primitive type " << type;
      break;
  }
}

Scalar IRParserFused::hlo_constant_to_scalar(
    xla::HloConstantInstruction const* constant) {
  xla::Shape const& shape = constant->shape();
  EXPECT_TRUE(shape.dimensions_size() == 0, "Constant must be scalar");

  Dtype dtype = Dtype::from_string(
      hlo_primitive_type_to_string(constant->shape().element_type()));

  Scalar scalar(
      mononn_engine::helpers::get_canonicalized_node_name(constant->name()),
      dtype, constant->literal().ToStringWithoutShapeOneline());

  return scalar;
}

TensorShape get_tensor_shape_from_xla_shape(const xla::Shape& shape) {
  std::vector<int> _shape;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      LOG(FATAL) << "Dynamic dimension " << i << " is not supported";
    }

    _shape.push_back((int)shape.dimensions(i));
  }

  if (_shape.empty()) {  // scalar
    _shape.push_back(1);
  }

  return TensorShape(_shape);
}

MemoryLayout get_memory_layout_from_xla_shape(const xla::Shape& shape) {
  std::vector<int> layout;
  std::vector<int> ordered_layout;

  if (shape.layout().tiles().length() != 0) LOG(FATAL) << "Unsupported layout";
  if (shape.layout().element_size_in_bits() != 0)
    LOG(FATAL) << "Unsupported layout";
  if (shape.layout().memory_space() != 0) LOG(FATAL) << "Unsupported layout";

  for (auto const& dim : shape.layout().minor_to_major()) {
    layout.push_back(dim);
  }

  // Convert from xla layout spec to onefuser spec
  for (int idx = 0; idx < (int)layout.size(); ++idx) {
    ordered_layout.push_back(std::find(layout.begin(), layout.end(), idx) -
                             layout.begin());
  }

  return MemoryLayout(ordered_layout);
}

Scalar get_reduction_init_value(xla::HloInstruction* instruction) {
  if (instruction->opcode() != xla::HloOpcode::kReduce &&
      instruction->opcode() != xla::HloOpcode::kReduceWindow) {
    LOG(FATAL) << "Unexpected inst type: "
               << xla::HloOpcodeString(instruction->opcode());
  }

  std::string node_name =
      mononn_engine::helpers::get_canonicalized_node_name(instruction->name());

  std::vector<Dtype> init_type_list;
  std::vector<std::string> init_value_list;

  for (int64_t param_idx = instruction->operand_count() / 2;
       param_idx < instruction->operand_count(); ++param_idx) {
    auto param_inst = instruction->operand(param_idx);
    if (param_inst->opcode() != xla::HloOpcode::kConstant) {
      LOG(FATAL) << "Operand " << param_inst->name() << " of reduction inst "
                 << instruction->name() << " is not constant";
    }

    Dtype type = Dtype::from_string(
        hlo_primitive_type_to_string(param_inst->shape().element_type()));
    std::string value =
        static_cast<const xla::HloConstantInstruction*>(param_inst)
            ->literal()
            .ToStringWithoutShapeOneline();

    if (value == "inf") {
      if (type == Dtype::FLOAT16) {
        value = "65504.0";
      } else if (type == Dtype::FLOAT32) {
        value = std::to_string(__FLT_MAX__);
      } else if (type == Dtype::INT32) {
        value = "2147483647";
      } else {
        LOG(FATAL) << "Unsupported type: " << type.to_string();
      }
    }

    if (value == "-inf") {
      if (type == Dtype::FLOAT16) {
        value = "-65504.0";
      } else if (type == Dtype::FLOAT32) {
        value = std::to_string(-__FLT_MAX__);
      } else if (type == Dtype::INT32) {
        value = "-2147483648";
      } else {
        LOG(FATAL) << "Unsupported type: " << type.to_string();
      }
    }

    init_type_list.push_back(type);
    init_value_list.push_back(value);
  }

  return Scalar(node_name + "_initializer", init_type_list, init_value_list);
}

//     Synchronization get_synchronization(Edge *edge) {
//         Op *ptr_src = edge->get_src().get();
//         Op *ptr_dst = edge->get_dst().get();

//         if (ptr_src->get_type() == OpType::constant ||
//             ptr_src->get_type() == OpType::parameter) {
//             return Synchronization::None;
//         }

//         if (ptr_dst->is_gemm() || ptr_dst->is_conv()) {
//             return Synchronization::Global;
//         }

//         if (ptr_src->is_gemm() || ptr_src->is_conv()) {
//             return Synchronization::Global;
//         }

//         if (ptr_src->is_cluster_reduce() && ptr_dst->is_cluster_reduce()) {
//             return Synchronization::ThreadBlock;
//             LOG(INFO) << "Sync can be optimized, however, additional
//             optimization depends on implementation which is not available at
//             parsing stage";
// //            if
// (ptr_src->as<ClusterReduce>()->get_reduce_node()->get_implementation()->as<ReduceImpl>()->get_tier()
// == LocalityTier::kT1 ||
// //
// ptr_dst->as<ClusterReduce>()->get_reduce_node()->get_implementation()->as<ReduceImpl>()->get_tier()
// == LocalityTier::kT1) {
// //
// //                return Synchronization::Warp;
// //            }
// //
// //            if
// (ptr_src->as<ClusterReduce>()->get_reduce_node()->get_implementation()->as<ReduceImpl>()->get_tier()
// == LocalityTier::kT2 ||
// //
// ptr_dst->as<ClusterReduce>()->get_reduce_node()->get_implementation()->as<ReduceImpl>()->get_tier()
// == LocalityTier::kT2) {
// //
// //                return Synchronization::ThreadBlock;
// //            }
//         }

//         return Synchronization::Global;
//     }

std::shared_ptr<Graph> IRParserFused::from_text(std::string text) {
  std::unique_ptr<xla::HloModule> ret =
      xla::ParseAndReturnUnverifiedModule(absl::string_view(text)).ValueOrDie();

  xla::HloModule* mod = ret.get();
  return IRParserFused::from_hlo_module(mod);
}

std::shared_ptr<Graph> IRParserFused::from_hlo_module_proto_file(
    std::string path_to_proto) {
  std::unique_ptr<xla::HloModuleProto> hlo_module_proto =
      std::make_unique<xla::HloModuleProto>();
  mononn_engine::helpers::load_proto_from_binary_file(hlo_module_proto.get(),
                                                      path_to_proto);

  xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  xla::HloModuleConfig module_config =
      xla::HloModule::CreateModuleConfigFromProto(*hlo_module_proto,
                                                  debug_options)
          .ValueOrDie();

  std::unique_ptr<xla::HloModule> hlo_module =
      xla::HloModule::CreateFromProto(*hlo_module_proto, module_config)
          .ValueOrDie();

  if (!Config::get()->dump_text_hlo_dir.empty()) {
    std::string dump_file = mononn_engine::helpers::Path::join(
        Config::get()->dump_text_hlo_dir, "hlo_module.txt");
    LOG(INFO) << "Dump parsed hlo module to file " << dump_file;
    std::ofstream ofs;
    ofs.open(dump_file);
    ofs << hlo_module->ToString();
    ofs.close();
  }

  return IRParserFused::from_hlo_module(hlo_module.get());
}

std::shared_ptr<Op> IRParserFused::get_node_from_hlo_instruction(
    Graph* graph, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
#define SWITCH_NODE(op_name, op_code, op_class_code, hlo_code)                 \
  case xla::HloOpcode::k##hlo_code: {                                          \
    std::shared_ptr<op_class_code> sketchy_node =                              \
        IRParserFused::get_sketchy_node<op_class_code>(graph, instruction);    \
    return initialize_##op_code(sketchy_node, instruction, fused_computation); \
  }

  switch (instruction->opcode()) {
    OP_TYPE_LIST(SWITCH_NODE)
    OP_TYPE_LIST_CLUSTER(SWITCH_NODE)
    default:
      LOG(FATAL) << "Unsupported op " << instruction->name() << " op code "
                 << instruction->opcode();
      break;
  }

#undef SWITCH_NODE
}

void IRParserFused::from_hlo_module_impl(Graph* graph,
                                         const xla::HloModule* hlo_module) {
  std::unordered_map<std::string, xla::HloComputation*> fused_computation;
  xla::HloComputation* entry_computation = hlo_module->entry_computation();

  for (xla::HloComputation* computation : hlo_module->computations()) {
    fused_computation[computation->name()] = computation;
  }

  for (xla::HloInstruction* instruction :
       entry_computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == xla::HloOpcode::kTuple) {
      // LOG(INFO) << "Skip tuple node " << instruction->name();
      continue;
    }

    std::shared_ptr<Op> node = IRParserFused::get_node_from_hlo_instruction(
        graph, instruction, &fused_computation);
    graph->add_node(node);

    for (xla::HloInstruction* input_instruction : instruction->operands()) {
      std::string input_node_name =
          mononn_engine::helpers::get_canonicalized_node_name(
              input_instruction->name());
      std::shared_ptr<Op> node_src = graph->get_node(input_node_name);
      std::shared_ptr<Edge> edge = std::make_shared<Edge>(node_src, node);

      graph->add_edge(edge);
    }

    std::string node_name = node->get_name();

    if (instruction->opcode() == xla::HloOpcode::kParameter) {
      graph->mark_as_input_node(node_name);
    }

    if (instruction->opcode() == xla::HloOpcode::kConstant) {
      if (node->as<Constant>()->is_scalar()) {
        graph->mark_as_extended_input_node(node_name);
      } else {
        graph->mark_as_input_node(node_name);
      }
    }

    if (instruction->opcode() == xla::HloOpcode::kIota) {
      graph->mark_as_extended_input_node(node_name);
    }
  }

  xla::HloInstruction* root_inst = entry_computation->root_instruction();

  if (root_inst->opcode() == xla::HloOpcode::kTuple) {
    for (auto const& operand : root_inst->operands()) {
      std::string operand_name =
          mononn_engine::helpers::get_canonicalized_node_name(operand->name());
      graph->mark_as_output_node(operand_name);
    }
  } else {
    std::string root_node_name =
        mononn_engine::helpers::get_canonicalized_node_name(root_inst->name());
    graph->mark_as_output_node(root_node_name);
  }

  graph->verify();

  // ClusterUtil::summary_graph(graph);
}

std::shared_ptr<Graph> IRParserFused::from_hlo_module(
    const xla::HloModule* hlo_module) {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();

  graph->set_graph_name(hlo_module->name());

  IRParserFused::from_hlo_module_impl(graph.get(), hlo_module);

  return graph;
}

std::unique_ptr<Graph> IRParserFused::from_hlo_module_unique(
    const xla::HloModule* hlo_module) {
  std::unique_ptr<Graph> graph = std::make_unique<Graph>();

  graph->set_graph_name(hlo_module->name());

  IRParserFused::from_hlo_module_impl(graph.get(), hlo_module);

  return std::move(graph);
}

std::shared_ptr<Graph> IRParserFused::from_file(std::string file_name) {
  std::ifstream file(file_name);
  std::stringstream text_stream;
  text_stream << file.rdbuf();
  return from_text(text_stream.str());
}

std::vector<std::shared_ptr<Op>> IRParserFused::get_operands(
    Graph* graph, xla::HloInstruction* instruction) {
  std::vector<std::shared_ptr<Op>> operands;

  for (xla::HloInstruction* operand_inst : instruction->operands()) {
    std::string operand_node_name =
        mononn_engine::helpers::get_canonicalized_node_name(
            operand_inst->name());

    if (!graph->has_node(operand_node_name)) {
      LOG(FATAL) << "Get operand " << operand_node_name << " for node "
                 << mononn_engine::helpers::get_canonicalized_node_name(
                        instruction->name())
                 << ", operand not in graph."
                 << " graph node list "
                 << mononn_engine::helpers::to_string(graph->get_node_list());
    }

    std::shared_ptr<Op> operand_node = graph->get_node(operand_node_name);

    operands.push_back(operand_node);
  }

  return operands;
}

std::vector<TensorSpec> IRParserFused::get_output_spec_list(
    xla::HloInstruction* instruction) {
  const xla::Shape& instruction_shape = instruction->shape();
  std::vector<TensorSpec> output_tensor_spec_list;

  if (instruction_shape.IsTuple()) {
    auto const& tuple_shapes = instruction_shape.tuple_shapes();

    for (auto const& shape : tuple_shapes) {
      if (shape.IsTuple()) {
        LOG(FATAL) << "Nested tuple shape is not supported. But inst "
                   << instruction->name() << " has shape "
                   << instruction_shape.ToString();
      }

      Dtype dtype = Dtype::from_string(
          hlo_primitive_type_to_string(shape.element_type()));

      TensorShape tensor_shape = get_tensor_shape_from_xla_shape(shape);
      MemoryLayout memory_layout = get_memory_layout_from_xla_shape(shape);
      output_tensor_spec_list.push_back(
          TensorSpec(dtype, tensor_shape, memory_layout));
    }
  } else {
    Dtype dtype = Dtype::from_string(
        hlo_primitive_type_to_string(instruction_shape.element_type()));
    TensorShape tensor_shape =
        get_tensor_shape_from_xla_shape(instruction_shape);
    MemoryLayout memory_layout =
        get_memory_layout_from_xla_shape(instruction_shape);
    output_tensor_spec_list.push_back(
        TensorSpec(dtype, tensor_shape, memory_layout));
  }

  return output_tensor_spec_list;
}

template <typename T>
struct OpFactory {
  static std::shared_ptr<T> make_shared(
      std::string node_name, std::vector<std::shared_ptr<Op>> operands,
      std::vector<TensorSpec> output_spec_list,
      xla::HloInstruction* instruction) {
    return std::make_shared<T>(node_name, operands, output_spec_list);
  }
};

template <>
struct OpFactory<ClusterOp> {
  static std::shared_ptr<ClusterOp> make_shared(
      std::string node_name, std::vector<std::shared_ptr<Op>> operands,
      std::vector<TensorSpec> output_spec_list,
      xla::HloInstruction* instruction) {
    xla::HloFusionInstruction* fusion_instruction =
        dynamic_cast<xla::HloFusionInstruction*>(instruction);

    bool root_reduce_node = false;

    if (fusion_instruction->fused_instructions_computation()
            ->root_instruction()
            ->opcode() == xla::HloOpcode::kReduce) {
      root_reduce_node = true;
    }

    bool found_reduce_node = false;

    for (auto const& inst : instruction->fused_instructions_computation()
                                ->MakeInstructionPostOrder()) {
      if (inst->opcode() == xla::HloOpcode::kReduce) {
        found_reduce_node = true;
      }
    }

    if (fusion_instruction->fusion_kind() ==
            xla::HloInstruction::FusionKind::kLoop &&
        !root_reduce_node) {
      return std::make_shared<ClusterElewise>(node_name, operands,
                                              output_spec_list);
    } else if (fusion_instruction->fusion_kind() ==
                   xla::HloInstruction::FusionKind::kInput &&
               !found_reduce_node) {
      LOG(WARNING) << "Fusion " << instruction->name() << " computation "
                   << instruction->fused_instructions_computation()->name()
                   << " is kInput fusion but no reduce node present. MonoNN "
                      "will treat it as loop fusion.";
      return std::make_shared<ClusterElewise>(node_name, operands,
                                              output_spec_list);
    } else if (fusion_instruction->fusion_kind() ==
                   xla::HloInstruction::FusionKind::kInput ||
               root_reduce_node) {
      return std::make_shared<ClusterReduce>(node_name, operands,
                                             output_spec_list);
    } else {
      LOG(FATAL) << "Unsupported fusion instruction " << instruction->name()
                 << " type " << fusion_instruction->fusion_kind();
    }
  }
};

template <typename T>
std::shared_ptr<T> IRParserFused::get_sketchy_node(
    Graph* graph, xla::HloInstruction* instruction) {
  std::string node_name =
      mononn_engine::helpers::get_canonicalized_node_name(instruction->name());
  std::vector<std::shared_ptr<Op>> operands = get_operands(graph, instruction);
  std::vector<TensorSpec> output_tensor_spec_list =
      get_output_spec_list(instruction);

  std::shared_ptr<T> node = OpFactory<T>::make_shared(
      node_name, operands, output_tensor_spec_list, instruction);
  auto option = xla::HloPrintOptions();
  option = option.set_print_large_constants(false);
  node->set_hlo_text(instruction->ToString(option));
  node->set_hlo_instruction_name(instruction->name());

  return node;
}

#define INSTANCIATE_TEMPLATE(op_name, op_code, op_class_code, ...) \
  template std::shared_ptr<op_class_code>                          \
      IRParserFused::get_sketchy_node<op_class_code>(              \
          Graph * graph, xla::HloInstruction * instruction);       \
  template struct OpFactory<op_class_code>;

OP_TYPE_LIST(INSTANCIATE_TEMPLATE)
OP_TYPE_LIST_CLUSTER(INSTANCIATE_TEMPLATE)

#undef INSTANCIATE_TEMPLATE

std::shared_ptr<Op> IRParserFused::initialize_abs(
    std::shared_ptr<Abs> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_add(
    std::shared_ptr<Add> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_bitcast(
    std::shared_ptr<Bitcast> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_broadcast(
    std::shared_ptr<Broadcast> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  std::vector<int> dimensions;

  for (auto const& d : instruction->dimensions()) {
    dimensions.push_back((int)d);
  }

  node->set_dimensions(dimensions);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_clamp(
    std::shared_ptr<Clamp> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_compare(
    std::shared_ptr<Compare> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  MathOp comparator;

  xla::ComparisonDirection direction =
      dynamic_cast<xla::HloCompareInstruction*>(instruction)->direction();
  if (direction == xla::ComparisonDirection::kEq) {
    comparator = MathOp::equal_to;
  } else if (direction == xla::ComparisonDirection::kNe) {
    comparator = MathOp::not_equal_to;
  } else if (direction == xla::ComparisonDirection::kGe) {
    comparator = MathOp::greater_equal_than;
  } else if (direction == xla::ComparisonDirection::kGt) {
    comparator = MathOp::greater_than;
  } else if (direction == xla::ComparisonDirection::kLe) {
    comparator = MathOp::less_equal_than;
  } else if (direction == xla::ComparisonDirection::kLt) {
    comparator = MathOp::less_than;
  } else {
    LOG(FATAL) << "Unsupported comparator " << (int)direction;
  }

  node->set_comparator(comparator);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_concatenate(
    std::shared_ptr<Concatenate> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloConcatenateInstruction* concatenate_inst =
      reinterpret_cast<xla::HloConcatenateInstruction*>(instruction);
  node->set_dimension((int)concatenate_inst->concatenate_dimension());

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_constant(
    std::shared_ptr<Constant> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  if (node->is_scalar()) {
    std::vector<int64_t> multi_index(instruction->shape().rank(), 0);
    std::string value = instruction->literal().GetAsString(multi_index);
    node->set_value(value);
  } else {
    auto TF_MONONN_ENABLED =
        mononn_engine::helpers::EnvVar::is_true("TF_MONONN_ENABLED");

    // When using TF->MonoNN bridge, there is no need to copy model weight.
    if (!TF_MONONN_ENABLED) {
      std::string node_name = node->get_name();
      Dtype dtype = node->get_output_spec(0).get_dtype();
      //            std::vector<int> shape =
      //            node->get_output_spec(0).get_shape().get_shape(); int rank =
      //            node->get_output_spec(0).rank(); int element_count =
      //            node->get_output_spec(0).element_count();

      std::vector<float> float_arr;
      std::vector<Eigen::half> half_arr;

      bool MONONN_STANDALONE_ENABLED =
          mononn_engine::helpers::EnvVar::is_true("MONONN_STANDALONE_ENABLED");

      if (MONONN_STANDALONE_ENABLED) {
        if (dtype == Dtype::FLOAT16) {
          auto data = instruction->literal().data<Eigen::half>();
          half_arr.insert(half_arr.end(), data.begin(), data.end());
          node->set_data_half(half_arr);
        } else if (dtype == Dtype::FLOAT32) {
          auto data = instruction->literal().data<float>();
          float_arr.insert(float_arr.end(), data.begin(), data.end());
          node->set_data_float(float_arr);
        }
      }
    }
    //            for (int idx = 0; idx < element_count; ++idx) {
    //                std::vector<int64_t> multi_index;
    //                int remaining_index = idx;
    //                for (auto r_iter = shape.rbegin(); r_iter != shape.rend();
    //                ++r_iter) {
    //                    multi_index.push_back((int64_t)remaining_index %
    //                    *r_iter); remaining_index /= *r_iter;
    //                }
    //
    //                std::reverse(multi_index.begin(), multi_index.end());
    //
    //                if (dtype == Dtype::FLOAT32) {
    //                    auto val =
    //                    instruction->literal().Get<float>(multi_index);
    //                    float_arr.push_back(val);
    //                } else if (dtype == Dtype::FLOAT16) {
    //                    auto val =
    //                    instruction->literal().Get<Eigen::half>(multi_index);
    //                    half_arr.push_back(val);
    //                } else {
    //                    LOG(FATAL) << "Unsupported dtype " <<
    //                    dtype.to_string();
    //                }
    //            }

    //            std::string save_file =
    //            mononn_engine::helpers::Path::join(Config::get()->build_path,
    //            node_name + ".npy"); if (dtype == Dtype::FLOAT32) {
    //                cnpy::npy_save(save_file, float_arr.data(), shape, "w");
    //            } else {
    //                cnpy::npy_save(save_file, half_arr.data(), shape, "w");
    //            }
  }

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_convert(
    std::shared_ptr<Convert> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_convolution(
    std::shared_ptr<Convolution> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  LOG(FATAL) << "Unimplemented";
}

std::shared_ptr<Op> IRParserFused::initialize_copy(
    std::shared_ptr<Copy> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_custom_call(
    std::shared_ptr<CustomCall> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloCustomCallInstruction* custom_call =
      dynamic_cast<xla::HloCustomCallInstruction*>(instruction);

  node->set_custom_call_target(custom_call->custom_call_target());
  node->set_backend_config_str(custom_call->opaque());

  if (node->is_conv()) {
    if (custom_call->feature_group_count() != 1) {
      LOG(FATAL) << "Group Convolution is not supported at this moment, get "
                    "feature group: "
                 << custom_call->feature_group_count() << " in node "
                 << node->get_name();
    }

    std::vector<int> filter_size;
    std::vector<int> filter_stride;
    std::vector<int> padding_low;
    std::vector<int> padding_high;

    int dim_size = instruction->window().dimensions_size();
    for (int idx = 0; idx < dim_size; ++idx) {
      filter_size.push_back((int)instruction->window().dimensions(idx).size());
      filter_stride.push_back(
          (int)instruction->window().dimensions(idx).stride());
      padding_low.push_back(
          (int)instruction->window().dimensions(idx).padding_low());
      padding_high.push_back(
          (int)instruction->window().dimensions(idx).padding_high());

      if (instruction->window().dimensions(idx).window_dilation() != 1) {
        LOG(FATAL) << "Dilated conv not supported for node: "
                   << node->get_name() << " dilation: "
                   << instruction->window().dimensions(idx).window_dilation();
      }
    }

    node->set_filter_size(filter_size);
    node->set_filter_stride(filter_stride);
    node->set_padding_low(padding_low);
    node->set_padding_high(padding_high);

    if (padding_low != padding_high) {
      std::stringstream error_msg;
      error_msg << "Only symmetric padding is supported.\n";
      error_msg << "padding low:";
      for (auto n : padding_low) {
        error_msg << " " << n;
      }

      error_msg << "\n";

      error_msg << "padding high:";
      for (auto n : padding_high) {
        error_msg << " " << n;
      }

      LOG(FATAL) << error_msg.str();
    }

    bool setted = false;

    for (auto const& user_inst : instruction->users()) {
      if (user_inst->opcode() != xla::HloOpcode::kGetTupleElement) {
        LOG(FATAL) << "Conv node have non-GTE user, conv node: "
                   << instruction->name()
                   << " Invalid user name: " << user_inst->name();
      }

      if (user_inst->tuple_index() == 0) {
        setted = true;
        node->set_conv_output_GTE_node_name(
            mononn_engine::helpers::get_canonicalized_node_name(
                user_inst->name()));
      }
    }

    if (!setted) {
      LOG(FATAL) << "GTE output node for conv: " << node->get_name()
                 << " not set.";
    }
  }

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_dynamic_slice(
    std::shared_ptr<DynamicSlice> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  LOG(DEBUG) << "Initialize dynamic slice: " << instruction->name()
             << " operand count " << instruction->operand_count();

  const std::vector<int64_t>& dynamic_slice_sizes =
      static_cast<xla::HloDynamicSliceInstruction*>(instruction)
          ->dynamic_slice_sizes();
  std::vector<int> node_dynamic_slice_size;

  for (auto const& size : dynamic_slice_sizes) {
    node_dynamic_slice_size.push_back(size);
  }

  node->set_dynamic_slice_sizes(node_dynamic_slice_size);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_dynamic_update_slice(
    std::shared_ptr<DynamicUpdateSlice> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_divide(
    std::shared_ptr<Divide> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_exp(
    std::shared_ptr<Exp> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_cluster(
    std::shared_ptr<ClusterOp> cluster_node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloFusionInstruction* fusion_instruction =
      dynamic_cast<xla::HloFusionInstruction*>(instruction);
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();

  bool has_dynamic_slice = false;
  bool has_dynamic_update_slice = false;
  bool has_gather = false;

  for (xla::HloInstruction* instruction :
       fusion_instruction->fused_instructions_computation()
           ->MakeInstructionPostOrder()) {
    if (instruction->opcode() == xla::HloOpcode::kTuple) {
      LOG(INFO) << "Skip tuple node " << instruction->name();
      continue;
    }

    std::shared_ptr<Op> node = IRParserFused::get_node_from_hlo_instruction(
        graph.get(), instruction, fused_computation);
    node->set_attribute(OpAttribute::initial_cluster_tag,
                        cluster_node->get_name());
    node->set_attribute(OpAttribute::sub_cluster_tag, cluster_node->get_name());
    node->set_attribute(OpAttribute::sub_cluster_type,
                        cluster_node->get_cluster_type().to_string());
    node->set_attribute(OpAttribute::initial_cluster_type,
                        cluster_node->get_cluster_type().to_string());

    graph->add_node(node);

    for (xla::HloInstruction* input_instruction : instruction->operands()) {
      std::string input_node_name =
          mononn_engine::helpers::get_canonicalized_node_name(
              input_instruction->name());
      std::shared_ptr<Op> node_src = graph->get_node(input_node_name);
      graph->add_edge(std::make_shared<Edge>(node_src, node));
    }

    if (instruction->opcode() == xla::HloOpcode::kGather) {
      // graph->add_control_edge(node->get_operand(1), node->get_operand(0));
      has_gather = true;
    }

    if (instruction->opcode() == xla::HloOpcode::kDynamicSlice) {
      has_dynamic_slice = true;
    }

    if (instruction->opcode() == xla::HloOpcode::kDynamicUpdateSlice) {
      has_dynamic_update_slice = true;
    }
  }

  for (xla::HloInstruction* instruction :
       fusion_instruction->fused_instructions_computation()->instructions()) {
    if (instruction->opcode() == xla::HloOpcode::kParameter) {
      std::string node_name =
          mononn_engine::helpers::get_canonicalized_node_name(
              instruction->name());
      graph->mark_as_input_node(node_name);
    }

    if (instruction->opcode() == xla::HloOpcode::kConstant ||
        instruction->opcode() == xla::HloOpcode::kIota) {
      std::string node_name =
          mononn_engine::helpers::get_canonicalized_node_name(
              instruction->name());
      graph->mark_as_extended_input_node(node_name);
    }
  }

  graph->sort_input_nodes();

  xla::HloInstruction* root_inst =
      fusion_instruction->fused_instructions_computation()->root_instruction();
  if (root_inst->opcode() == xla::HloOpcode::kTuple) {
    for (auto const& operand : root_inst->operands()) {
      std::string operand_name =
          mononn_engine::helpers::get_canonicalized_node_name(operand->name());
      graph->mark_as_output_node(operand_name);
    }
  } else {
    std::string root_node_name =
        mononn_engine::helpers::get_canonicalized_node_name(root_inst->name());
    graph->mark_as_output_node(root_node_name);
  }

  // Add control dependencies.
  if (has_dynamic_slice || has_dynamic_update_slice || has_gather) {
    graph->build_transitive_closure();

    for (auto const& node_name : graph->get_node_list()) {
      auto const node = graph->get_node(node_name);

      if (node->get_type() == OpType::dynamic_slice) {
        for (int operand_id = 1; operand_id < node->get_operand_count();
             ++operand_id) {
          for (auto const& input_node_name : graph->get_input_nodes()) {
            if (graph->topology_before(input_node_name,
                                       node->get_operand(0)->get_name())) {
              graph->add_control_edge(node->get_operand(operand_id)->get_name(),
                                      input_node_name);
            }
          }
        }
      }

      if (node->get_type() == OpType::dynamic_update_slice) {
        if (!graph->is_output_node(node_name)) {
          LOG(FATAL) << "DynamicUpdateSlice node " << node_name
                     << " is not output node in cluster "
                     << cluster_node->get_name();
        }

        for (int operand_id = 2; operand_id < node->get_operand_count();
             ++operand_id) {
          for (auto const& input_node_name : graph->get_input_nodes()) {
            if (graph->topology_before(input_node_name,
                                       node->get_operand(0)->get_name()) ||
                graph->topology_before(input_node_name,
                                       node->get_operand(1)->get_name())) {
              graph->add_control_edge(node->get_operand(operand_id)->get_name(),
                                      input_node_name);
            }
          }
        }
      }

      if (node->get_type() == OpType::gather) {
        for (auto const& input_node_name : graph->get_input_nodes()) {
          if (graph->topology_before(input_node_name,
                                     node->get_operand(0)->get_name()) ||
              input_node_name == node->get_operand(0)->get_name()) {
            graph->add_control_edge(node->get_operand(1)->get_name(),
                                    input_node_name);
          }
        }
      }
    }
  }

  graph->verify();

  cluster_node->set_graph(graph);

  cluster_node->add_hlo_instruction_name(instruction->name());

  // if (cluster_node->is_cluster_reduce()) {
  // cluster_node->append_sub_cluster_tag(cluster_node->get_name());
  // cluster_node->append_sub_cluster_type(cluster_node->get_cluster_type().to_string());
  // }

  return std::static_pointer_cast<Op>(cluster_node);
}

std::shared_ptr<Op> IRParserFused::initialize_gather(
    std::shared_ptr<Gather> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloGatherInstruction* gather_inst =
      dynamic_cast<xla::HloGatherInstruction*>(instruction);
  int index_vector_dim;
  std::vector<int> offset_dims;
  std::vector<int> slice_sizes;
  std::vector<int> collapsed_slice_dims;
  std::vector<int> start_index_map;
  bool indices_are_sorted;
  bool unique_indices;

  const auto& gather_dimension_numbers =
      gather_inst->gather_dimension_numbers();

  index_vector_dim = (int)gather_dimension_numbers.index_vector_dim();

  for (auto const& d : gather_dimension_numbers.offset_dims()) {
    offset_dims.push_back((int)d);
  }

  for (auto const& d : gather_inst->gather_slice_sizes()) {
    slice_sizes.push_back((int)d);
  }

  for (auto const& d : gather_dimension_numbers.collapsed_slice_dims()) {
    collapsed_slice_dims.push_back((int)d);
  }

  for (auto const& d : gather_dimension_numbers.start_index_map()) {
    start_index_map.push_back((int)d);
  }

  indices_are_sorted = gather_inst->indices_are_sorted();

  // Somehow HloGatherInstruction do not implemented this method
  // unique_indices = gather_inst->unique_indices();

  node->set_index_vector_dim(index_vector_dim);
  node->set_offset_dims(offset_dims);
  node->set_slice_sizes(slice_sizes);
  node->set_collapsed_slice_dims(collapsed_slice_dims);
  node->set_start_index_map(start_index_map);
  node->set_indices_are_sorted(indices_are_sorted);
  node->set_unique_indices(unique_indices);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_get_tuple_element(
    std::shared_ptr<GetTupleElement> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloGetTupleElementInstruction* tuple_inst =
      static_cast<xla::HloGetTupleElementInstruction*>(instruction);

  node->set_tuple_index((int)tuple_inst->tuple_index());

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_iota(
    std::shared_ptr<Iota> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  int iota_dimension = (int)dynamic_cast<xla::HloIotaInstruction*>(instruction)
                           ->iota_dimension();

  node->set_iota_dimension(iota_dimension);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_log(
    std::shared_ptr<Log> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_maximum(
    std::shared_ptr<Maximum> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_minimum(
    std::shared_ptr<Minimum> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_multiply(
    std::shared_ptr<Multiply> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_negate(
    std::shared_ptr<Negate> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_sign(
    std::shared_ptr<Sign> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_pad(
    std::shared_ptr<Pad> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  std::vector<int> padding_low;
  std::vector<int> padding_high;

  xla::HloPadInstruction* padding =
      dynamic_cast<xla::HloPadInstruction*>(instruction);

  // Scalar padding_value =
  //     hlo_constant_to_scalar(dynamic_cast<xla::HloConstantInstruction
  //     const*>(
  //         padding->padding_value()));

  for (auto const& config : padding->padding_config().dimensions()) {
    EXPECT_TRUE(config.interior_padding() == 0,
                "Interior padding not supported");
    EXPECT_TRUE(config.edge_padding_high() >= 0, "Padding must non-negative");
    EXPECT_TRUE(config.edge_padding_low() >= 0, "Padding must non-negative");

    padding_low.push_back((int)config.edge_padding_low());
    padding_high.push_back((int)config.edge_padding_high());
  }

  node->set_padding_low(padding_low);
  node->set_padding_high(padding_high);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_parameter(
    std::shared_ptr<Parameter> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloParameterInstruction* parameter =
      static_cast<xla::HloParameterInstruction*>(instruction);
  node->set_parameter_number((int)parameter->parameter_number());

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_reduce(
    std::shared_ptr<Reduce> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  // EXPECT_TRUE(instruction->operand_count() == 2,
  // mononn_engine::helpers::string_format("Reduce node %s have %d operands",
  // node->get_name().c_str(), instruction->operand_count()).c_str());
  EXPECT_TRUE(instruction->dimensions().size() == 1,
              "Multiple dimension not supported yet");

  node->set_dimension((int)instruction->dimensions()[0]);

  // xla::HloInstruction *constant = dynamic_cast<xla::HloReduceInstruction
  // *>(instruction)->init_values()[0];

  // Scalar init_value =
  // hlo_constant_to_scalar(dynamic_cast<xla::HloConstantInstruction
  // *>(constant)); node->set_init_value(init_value); Reducer reducer;

  // if (instruction->to_apply()->root_instruction()->opcode() ==
  // xla::HloOpcode::kAdd) {
  //     reducer = Reducer::Sum;
  // } else if (instruction->to_apply()->root_instruction()->opcode() ==
  // xla::HloOpcode::kMaximum) {
  //     reducer = Reducer::Max;
  // } else if (instruction->to_apply()->root_instruction()->opcode() ==
  // xla::HloOpcode::kMinimum) {
  //     reducer = Reducer::Min;
  // }  else {
  //     LOG(FATAL) << "Unsupported reduction operation " <<
  //     instruction->to_apply()->root_instruction()->name() << std::endl;
  // }

  // node->set_reducer(reducer);

  ReductionFunctorGenerator::Registry()->add_generator(node->get_name(),
                                                       instruction->to_apply());

  node->set_reduction_functor_generator(
      ReductionFunctorGenerator::Registry()->get_generator(
          instruction->to_apply()->name()));

  node->set_init_value(get_reduction_init_value(instruction));

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_reduce_window(
    std::shared_ptr<ReduceWindow> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  EXPECT_TRUE(instruction->operand_count() == 2,
              mononn_engine::helpers::string_format(
                  "ReduceWindow node %s have %d operands. Expect 2.",
                  node->get_name().c_str(), instruction->operand_count())
                  .c_str());

  ReductionFunctorGenerator::Registry()->add_generator(node->get_name(),
                                                       instruction->to_apply());

  node->set_reduction_functor_generator(
      ReductionFunctorGenerator::Registry()->get_generator(
          instruction->to_apply()->name()));

  std::vector<int> filter_size;
  std::vector<int> filter_stride;
  std::vector<int> padding_low;
  std::vector<int> padding_high;

  int dim_size = instruction->window().dimensions_size();
  for (int idx = 0; idx < dim_size; ++idx) {
    filter_size.push_back((int)instruction->window().dimensions(idx).size());
    filter_stride.push_back(
        (int)instruction->window().dimensions(idx).stride());
    padding_low.push_back(
        (int)instruction->window().dimensions(idx).padding_low());
    padding_high.push_back(
        (int)instruction->window().dimensions(idx).padding_high());

    if (instruction->window().dimensions(idx).window_dilation() != 1) {
      LOG(FATAL) << "Dilated conv not supported for node: " << node->get_name()
                 << " dilation: "
                 << instruction->window().dimensions(idx).window_dilation();
    }
  }

  for (int idx = 0; idx < dim_size; ++idx) {
    if (filter_stride[idx] == 0) {
      if (filter_size[idx] != 0) {
        LOG(FATAL) << "Invalid filter stride and size: idx " << idx
                   << " stride " << filter_stride[idx] << " size "
                   << filter_size[idx] << " node name " << node->get_name();
      }

      if (padding_low[idx] != 0) {
        LOG(FATAL) << "Invalid filter stride and padding low: idx " << idx
                   << " stride " << filter_stride[idx] << " size "
                   << padding_low[idx] << node->get_name();
      }

      if (padding_high[idx] != 0) {
        LOG(FATAL) << "Invalid filter stride and padding high: idx " << idx
                   << " stride " << filter_stride[idx] << " size "
                   << padding_high[idx] << node->get_name();
      }
    }
  }

  node->set_filter_size(filter_size);
  node->set_filter_stride(filter_stride);
  node->set_padding_low(padding_low);
  node->set_padding_high(padding_high);

  node->set_init_value(get_reduction_init_value(instruction));

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_reshape(
    std::shared_ptr<Reshape> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_rsqrt(
    std::shared_ptr<Rsqrt> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_select(
    std::shared_ptr<Select> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_slice(
    std::shared_ptr<Slice> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloSliceInstruction* slice_inst =
      static_cast<xla::HloSliceInstruction*>(instruction);

  std::vector<int> slice_starts;
  std::vector<int> slice_limits;
  std::vector<int> slice_strides;

  int rank = node->get_output_spec(0).get_shape().rank();

  for (int idx = 0; idx < rank; ++idx) {
    slice_starts.push_back((int)slice_inst->slice_starts(idx));
    slice_limits.push_back((int)slice_inst->slice_limits(idx));
    slice_strides.push_back((int)slice_inst->slice_strides(idx));
  }

  EXPECT_TRUE(
      std::all_of(slice_strides.begin(), slice_strides.end(),
                  [&](int stride) -> bool { return stride == 1; }),
      "Slice stride should be one");  // vectorizer and index tracer should be
                                      // modified accordingly for non-one stride
                                      // Non-one stride will also impact
                                      // vectorization.

  node->set_slice_starts(slice_starts);
  node->set_slice_limits(slice_limits);
  node->set_slice_strides(slice_strides);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_subtract(
    std::shared_ptr<Subtract> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_tanh(
    std::shared_ptr<Tanh> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_transpose(
    std::shared_ptr<Transpose> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  xla::HloTransposeInstruction* transpose_impl =
      static_cast<xla::HloTransposeInstruction*>(instruction);
  std::vector<int> perm;

  for (auto const& d : transpose_impl->dimensions()) {
    perm.push_back((int)d);
  }

  node->set_permute(perm);

  return std::static_pointer_cast<Op>(node);
}

std::shared_ptr<Op> IRParserFused::initialize_tuple(
    std::shared_ptr<Tuple> node, xla::HloInstruction* instruction,
    std::unordered_map<std::string, xla::HloComputation*>* fused_computation) {
  return std::static_pointer_cast<Op>(node);
}
}  // namespace parser
}  // namespace mononn_engine