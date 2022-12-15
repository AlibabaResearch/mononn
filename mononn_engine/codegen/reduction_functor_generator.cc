#include <sstream>

#include "mononn_engine/codegen/reduction_functor_generator.h"
#include "mononn_engine/helpers/string_helpers.h"

namespace mononn_engine {
namespace codegen {
    const std::string &ReductionFunctorGenerator::id() const {
        return this->computation_name;
    }

    const std::string &ReductionFunctorGenerator::type_name() const {
        return this->functor_type_name;
    }

    const std::string &ReductionFunctorGenerator::instance_name() const {
        return this->functor_instance_name;
    }

    std::string ReductionFunctorGenerator::generate_functor_definition() const {
        return this->functor_definition;
    }

    ReductionFunctorGeneratorRegistry *ReductionFunctorGenerator::Registry() {
        if (!ReductionFunctorGenerator::registry) {
            ReductionFunctorGenerator::registry = std::make_unique<ReductionFunctorGeneratorRegistry>();
        }

        return ReductionFunctorGenerator::registry.get();
    }

    std::string hlo_primitive_type_to_string(xla::PrimitiveType type) {
        switch (type) {
            case xla::PRED: return "bool";
            case xla::S8:   return "int8_t";
            case xla::S16:  return "int16_t";
            case xla::S32:  return "int32_t";
            case xla::S64:  return "int64_t";
            case xla::U8:   return "uint8_t";
            case xla::U16:  return "uint16_t";
            case xla::U32:  return "uint32_t";
            case xla::U64:  return "uint64_t";
            case xla::F16:  return "half";
            case xla::F32:  return "float";
            default:
                LOG(FATAL) << "Unsupported hlo primitive type " << type;
                break;
        }
    }

    std::string hlo_shape_to_type_string(const xla::Shape &shape) {
        if (shape.IsTuple()) {
            std::vector<std::string> type_list;

            for (auto const &sub_shape : shape.tuple_shapes()) {
                if (sub_shape.IsTuple()) {
                    LOG(FATAL) << shape.ToString() << " not supported.";
                }

                type_list.push_back( hlo_primitive_type_to_string(sub_shape.element_type()));
            }

            return "cuda::std::tuple<" + mononn_engine::helpers::join(", ", type_list) + ">";
        } else {
            return hlo_primitive_type_to_string(shape.element_type());
        }
    }

    std::string get_inst_definition(const xla::HloInstruction *inst, const std::string &value) {
        return hlo_shape_to_type_string(inst->shape()) + " " + mononn_engine::helpers::get_canonicalized_node_name(inst->name()) + " = " + value;
    }

    std::string emit_parameter(const xla::HloInstruction *inst) {
        return "";
    }

    std::string emit_fusion(const xla::HloInstruction *inst) {
        std::string invocaiton = mononn_engine::helpers::get_canonicalized_node_name(inst->fused_instructions_computation()->name());
        std::vector<std::string> param_list;

        for (auto const &operand : inst->operands()) {
            param_list.push_back(mononn_engine::helpers::get_canonicalized_node_name(operand->name()));
        }

        invocaiton += "(" + mononn_engine::helpers::join(", ", param_list) + ")";

        return get_inst_definition(inst, invocaiton);
    }

    std::string emit_get_tuple_element(const xla::HloInstruction *inst) {
        std::string invocaiton = "cuda::std::get<" + std::to_string(inst->operand_count()) + ">(" 
                + mononn_engine::helpers::get_canonicalized_node_name(inst->operand(0)->name()) + ")";
        return get_inst_definition(inst, invocaiton);
    }

    std::string emit_compare(const xla::HloInstruction *inst) {
        std::string cmp;
        switch (inst->comparison_direction()) {
            case xla::ComparisonDirection::kEq:
                cmp = " == ";
                break;
            case xla::ComparisonDirection::kGe:
                cmp = " >= ";
                break;
            case xla::ComparisonDirection::kGt:
                cmp = " > ";
                break;
            case xla::ComparisonDirection::kLe:
                cmp = " <= ";
                break;
            case xla::ComparisonDirection::kLt:
                cmp = " < ";
                break;
            case xla::ComparisonDirection::kNe:
                cmp = " != ";
                break;
            default:
                LOG(FATAL) << "Not supported comparator: " << (int)inst->comparison_direction();
        }

        std::string param_a = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(0)->name());
        std::string param_b = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(1)->name());

        return get_inst_definition(inst, param_a + cmp + param_b);
    }

    std::string emit_elewise_binary(const xla::HloInstruction *inst) {
        std::string operand_a = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(0)->name());
        std::string operand_b = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(1)->name());

        std::string value;

        switch (inst->opcode()) {
            case xla::HloOpcode::kAdd:
                value = operand_a + " + " + operand_b;
                break;
            case xla::HloOpcode::kMaximum:
                value = mononn_engine::helpers::string_format("%s > %s ? %s : %s", operand_a.c_str(), operand_b.c_str(), operand_a.c_str(), operand_b.c_str());
                break;
            case xla::HloOpcode::kMinimum:
                value = mononn_engine::helpers::string_format("%s < %s ? %s : %s", operand_a.c_str(), operand_b.c_str(), operand_a.c_str(), operand_b.c_str());
                break;
            case xla::HloOpcode::kAnd:
                value = operand_a + " && " + operand_b;
                break;
            default:
                LOG(FATAL) << "Unsupported opcode " << xla::HloOpcodeString(inst->opcode()) 
                        << " of instruciton " << inst->name() << " not supported.";
        }

        return get_inst_definition(inst, value);
    }

    std::string emit_select(const xla::HloInstruction *inst) {
        std::string pred = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(0)->name());
        std::string on_true = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(1)->name());
        std::string on_false = mononn_engine::helpers::get_canonicalized_node_name(inst->operand(2)->name());

        return get_inst_definition(inst, pred + " ? " + on_true + " : " + on_false);
    }

    std::string emit_tuple(const xla::HloInstruction *inst) {
        std::vector<std::string> tuple_elements;

        for (auto const &operand : inst->operands()) {
            tuple_elements.push_back(mononn_engine::helpers::get_canonicalized_node_name(operand->name()));
        }

        return get_inst_definition(inst, "cuda::std::make_tuple(" + mononn_engine::helpers::join(", ", tuple_elements) + ")");
    }

    std::string generate_fused_computation(const xla::HloComputation *fused_computation) {
        std::stringstream computation_definition;

        computation_definition << "__device__ __forceinline__\n";
        computation_definition << hlo_shape_to_type_string(fused_computation->root_instruction()->shape()) 
            << mononn_engine::helpers::get_canonicalized_node_name(fused_computation->name()) << "(\n";
        
        for (int64_t param_idx = 0; param_idx < fused_computation->num_parameters(); ++param_idx) {
            auto param_inst = fused_computation->parameter_instruction(param_idx);
            computation_definition << "const " << hlo_primitive_type_to_string(param_inst->shape().element_type()) <<" &"
                << mononn_engine::helpers::get_canonicalized_node_name(param_inst->name());

            if (param_idx == fused_computation->num_parameters() - 1) {
                computation_definition << ") {\n";
            } else {
                computation_definition << ",\n";
            }
        }

        for (auto const &inst : fused_computation->MakeInstructionPostOrder()) {
            switch (inst->opcode()) {
                case xla::HloOpcode::kParameter: {
                    break;
                }
                case xla::HloOpcode::kGetTupleElement: {
                    computation_definition << emit_get_tuple_element(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kTuple: {
                    computation_definition << emit_tuple(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kSelect: {
                    computation_definition << emit_select(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kCompare: {
                    computation_definition << emit_compare(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kMaximum:
                case xla::HloOpcode::kMinimum:
                case xla::HloOpcode::kAnd:
                case xla::HloOpcode::kAdd: {
                    computation_definition << emit_elewise_binary(inst) << ";\n";
                    break;
                }
                default:
                    LOG(FATAL) << "Unsupported opcode " << xla::HloOpcodeString(inst->opcode()) 
                        << " of instruciton " << inst->name() << " not supported.";
            }
        }

        computation_definition << "return " << mononn_engine::helpers::get_canonicalized_node_name(fused_computation->root_instruction()->name()) << ";\n";
        computation_definition << "}\n";

        return computation_definition.str();
    }

    void ReductionFunctorGeneratorRegistry::add_generator(const std::string &reduction_node_name, const xla::HloComputation *computation) {
        if (this->registry.count(computation->name())) {
            return;
        }

        std::string functor_type_name = mononn_engine::helpers::get_canonicalized_node_name(computation->name());
        std::string functor_instance_name = functor_type_name + "__";
        
        std::stringstream functor_definition;

        for (auto const &inst : computation->MakeInstructionPostOrder()) {
            if (inst->opcode() == xla::HloOpcode::kFusion) {
                if (inst->fusion_kind() != xla::HloInstruction::FusionKind::kLoop) {
                    LOG(FATAL) << "Fusion inst: " << inst->name() << " inside computation " << computation->name() << " is not loop fusion.";
                }

                functor_definition << generate_fused_computation(inst->fused_instructions_computation());
            }
        }

        functor_definition << "struct " << functor_type_name << " {\n";

        functor_definition << "__device__ __forceinline__\n";
        functor_definition << hlo_shape_to_type_string(computation->root_instruction()->shape()) << " operator () ";

        std::vector<std::string> init_value_type;
        std::vector<std::string> new_value_type;
        // parameter order remapping from hlo computation parameter to functor parameter. 
        for (int64_t param_idx = 0; param_idx < computation->num_parameters(); ++param_idx) {
            auto param_inst = computation->parameter_instruction(param_idx);

            if (param_idx < computation->num_parameters() / 2) {
                new_value_type.push_back(hlo_primitive_type_to_string(param_inst->shape().element_type()));
            } else {
                init_value_type.push_back(hlo_primitive_type_to_string(param_inst->shape().element_type()));
            }
        }

        if (init_value_type != new_value_type) {
            LOG(FATAL) << "Value type not match";
        }

        if (init_value_type.empty()) {
            LOG(FATAL) << "Empty value type.";
        }

        functor_definition << "(const cuda::std::tuple<" << mononn_engine::helpers::join(", ", init_value_type) << "> &a, "
            << "const cuda::std::tuple<" << mononn_engine::helpers::join(", ", init_value_type) << "> &b) const {\n";

        // computation->Accept(DfsHloVisitorBase<HloInstructionPtr> *visitor)
        for (auto const &inst : computation->MakeInstructionPostOrder()) {
            switch (inst->opcode()) {
                case xla::HloOpcode::kParameter: {
                    int64_t param_idx = inst->parameter_number();
                    if (param_idx < computation->num_parameters() / 2) {
                        functor_definition << get_inst_definition(inst, "cuda::std::get<" + std::to_string(param_idx) + ">(a)") << ";\n";
                    } else {
                        functor_definition << get_inst_definition(inst, "cuda::std::get<" + std::to_string(param_idx - (computation->num_parameters() / 2)) + ">(b)") << ";\n";
                    }
                    break;
                }
                case xla::HloOpcode::kFusion: {
                    functor_definition << emit_fusion(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kGetTupleElement: {
                    functor_definition << emit_get_tuple_element(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kTuple: {
                    functor_definition << emit_tuple(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kSelect: {
                    functor_definition << emit_select(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kCompare: {
                    functor_definition << emit_compare(inst) << ";\n";
                    break;
                }
                case xla::HloOpcode::kMaximum:
                case xla::HloOpcode::kMinimum:
                case xla::HloOpcode::kAnd:
                case xla::HloOpcode::kAdd: {
                    functor_definition << emit_elewise_binary(inst) << ";\n";
                    break;
                }
                default:
                    LOG(FATAL) << "Unsupported opcode " << xla::HloOpcodeString(inst->opcode()) 
                        << " of instruciton " << inst->name() << " not supported.";
            }
        }

        functor_definition << "return " << mononn_engine::helpers::get_canonicalized_node_name(computation->root_instruction()->name()) << ";\n";
        functor_definition << "}\n";

        if (computation->num_parameters() == 2) { // Scalar reduction, not tuple
            functor_definition << "__device__ __forceinline__\n";
            functor_definition << hlo_shape_to_type_string(computation->root_instruction()->shape()) << " operator () ";
            functor_definition << "(const " << init_value_type[0] << " &a, " << "const " << init_value_type[0] << " &b) const {\n";
            functor_definition << "return (*this)(cuda::std::make_tuple(a), cuda::std::make_tuple(b));\n";
            functor_definition << "}\n";
        }

        functor_definition << "} __device__ " << functor_instance_name << ";\n";

        this->registry[computation->name()] 
            = std::move(std::make_unique<ReductionFunctorGenerator>(reduction_node_name, computation->name(), functor_type_name, functor_instance_name, functor_definition.str()));
    }

    const std::unordered_map<ReductionFunctorGeneratorRegistry::ComputationName, 
        std::unique_ptr<ReductionFunctorGenerator>>& ReductionFunctorGeneratorRegistry::get_generator() const {
        
        return this->registry;
    }

    const ReductionFunctorGenerator *ReductionFunctorGeneratorRegistry::get_generator(const std::string &computaiton_name) const {
        if (!this->registry.count(computaiton_name)) {
            LOG(FATAL) << computaiton_name << " not in registry.";
        }

        return this->registry.at(computaiton_name).get();
    }

    thread_local std::unique_ptr<ReductionFunctorGeneratorRegistry> ReductionFunctorGenerator::registry = nullptr;
}
}