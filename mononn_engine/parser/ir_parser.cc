#include "mononn_engine/core/tensor/dtype.h"
#include "mononn_engine/parser/ir_parser.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "mononn_engine/core/op/all_operators.h"
#include "mononn_engine/helpers/macros.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "mononn_engine/core/op_impl/reducer.h"
#include "mononn_engine/core/tensor/math_op.h"
#include "tensorflow/core/platform/statusor.h"

namespace mononn_engine {
namespace parser {
    using Graph = mononn_engine::core::graph::Graph;
    using Op = mononn_engine::core::op::Op;
    using Edge = mononn_engine::core::edge::Edge<Op>;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using MemoryLayout = mononn_engine::core::tensor::MemoryLayout;
    using TensorSpec = mononn_engine::core::tensor::TensorSpec;
    using Dtype = mononn_engine::core::tensor::Dtype;
    using Reducer = mononn_engine::core::op_impl::Reducer;
    using MathOp = mononn_engine::core::tensor::MathOp;
    using Scalar = mononn_engine::core::tensor::Scalar;

    #define JOIN_USING(x, y) x::y

    #define USING_OP(op_name, op_code, op_class_code, ...) \
    using op_class_code = JOIN_USING(mononn_engine::core::op, op_class_code);

    OP_TYPE_LIST(USING_OP)

    #undef USING_OP
    #undef JOIN_USING

    
    std::string hlo_primitive_type_to_string(xla::PrimitiveType type) {
        switch (type) {
            case xla::PRED: return "bool";
            case xla::S8: return "int8";
            case xla::S16: return "int16";
            case xla::S32: return "int32";
            case xla::S64: return "int64";
            case xla::U16: return "uint16";
            case xla::U32: return "uint32";
            case xla::U64: return "uint64";
            case xla::F16: return "float16";
            case xla::F32: return "float32";
            case xla::F64: return "float64";
            default:
                LOG(FATAL) << "Unsupported hlo primitive type " << type;
                break;
        }
    }

    Scalar hlo_constant_to_scalar(xla::HloConstantInstruction const *constant) {
        xla::Shape const &shape = constant->shape();
        EXPECT_TRUE(shape.dimensions_size() == 0, "Constant must be scalar");

        Dtype dtype = Dtype::from_string(hlo_primitive_type_to_string(constant->shape().element_type()));

        Scalar scalar(constant->name(), dtype, constant->literal().ToStringWithoutShapeOneline());

        return scalar;
    }

    TensorShape get_tensor_shape_from_xla_shape(const xla::Shape &shape) {
        std::vector<int> _shape;
        for (int i = 0; i < shape.dimensions_size(); ++i) {
            if (shape.is_dynamic_dimension(i)) {
                LOG(FATAL) << "Dynamic dimension " << i << " is not supported";
            }

            _shape.push_back((int)shape.dimensions(i));
        }

        return TensorShape(_shape);
    }

    MemoryLayout get_memory_layout_from_xla_shape(const xla::Shape &shape) {
        std::vector<int> layout;
        std::vector<int> ordered_layout;

        if (shape.layout().tiles().length() != 0) LOG(FATAL) << "Unsupported layout";
        if (shape.layout().element_size_in_bits() != 0) LOG(FATAL) << "Unsupported layout";
        if (shape.layout().memory_space() != 0) LOG(FATAL) << "Unsupported layout";

        for (auto const &dim : shape.layout().minor_to_major()) {
            layout.push_back(dim);
        }

        for (int idx = 0; idx < (int)layout.size(); ++idx) {
            ordered_layout.push_back(std::find(layout.begin(), layout.end(), idx) - layout.begin());
        }

        return MemoryLayout(ordered_layout);
    }

    std::shared_ptr<Graph> IRParser::from_text(std::string text) {
        std::shared_ptr<Graph> graph = std::make_shared<Graph>();
        std::unordered_map<std::string, xla::HloComputation *> fused_computation;

        tensorflow::StatusOr<std::unique_ptr<xla::HloModule>> ret = 
                            xla::ParseAndReturnUnverifiedModule(absl::string_view(text));

        xla::HloModule *mod = ret->get();
        xla::HloComputation *entry_computation = ret->get()->entry_computation();

        for (xla::HloComputation *computation : mod->computations()) {
            fused_computation[computation->name()] = computation;
        }

        for (xla::HloInstruction *instruction : entry_computation->instructions()) {
            std::shared_ptr<Op> node = get_node_from_hlo_instruction(graph, instruction, &fused_computation);

            graph->add_node(node);

            for (xla::HloInstruction *input_instruction : instruction->operands()) {
                std::shared_ptr<Op> node_src = graph->get_node(input_instruction->name());
                graph->add_edge(std::make_shared<Edge>(node_src, node));
            }
        }

        // for (xla::HloInstruction *instruction : entry_computation->parameter_instructions()) {
        //     graph->mark_as_input_node(instruction->name());
        // }

        for (xla::HloInstruction *instruction : entry_computation->instructions()) {
            if (instruction->opcode() == xla::HloOpcode::kParameter ||
                instruction->opcode() == xla::HloOpcode::kConstant || 
                instruction->opcode() == xla::HloOpcode::kIota) {
                graph->mark_as_input_node(instruction->name());
            }
        }

        graph->mark_as_output_node(entry_computation->root_instruction()->name());

        graph->verify();

        LOG(INFO) << "Graph summary:";
        LOG(INFO) << graph->summary();

        return graph;
    }

    std::shared_ptr<Graph> IRParser::from_file_stream(std::ifstream &file_stream) {
        std::stringstream text_stream;
        text_stream << file_stream.rdbuf();
        return from_text(text_stream.str());
    }

    template<typename T>
    std::shared_ptr<T> IRParser::get_sketchy_node(
        std::shared_ptr<Graph> graph,
        xla::HloInstruction *instruction
    ) {
        std::string node_name = instruction->name();
        std::vector<std::shared_ptr<Op>> operands;
        
        for (xla::HloInstruction *operand_inst : instruction->operands()) {
            std::string operand_node_name = operand_inst->name();
            std::shared_ptr<Op> operand_node = graph->get_node(operand_node_name);

            operands.push_back(operand_node);
        }

        const xla::Shape &shape = instruction->shape();

        Dtype dtype;
        TensorShape tensor_shape;
        MemoryLayout memory_layout;

        if (shape.IsTuple()) {
            auto const &tuple_shapes = shape.tuple_shapes();
            dtype = Dtype::from_string(hlo_primitive_type_to_string(tuple_shapes[0].element_type()));
            tensor_shape = get_tensor_shape_from_xla_shape(tuple_shapes[0]);
            memory_layout = get_memory_layout_from_xla_shape(tuple_shapes[0]);
        } else {
            dtype = Dtype::from_string(hlo_primitive_type_to_string(shape.element_type()));
            tensor_shape = get_tensor_shape_from_xla_shape(shape);
            memory_layout = get_memory_layout_from_xla_shape(shape);
        }

        TensorSpec output_tensor_spec(dtype, tensor_shape, memory_layout);
        std::vector<TensorSpec> output_tensor_spec_list = {output_tensor_spec};

        if (shape.IsTuple()) {
            auto const &tuple_shapes = shape.tuple_shapes();
            for (int idx = 1; idx < (int)tuple_shapes.size(); ++idx) {
                const xla::Shape &elem_shape = tuple_shapes[idx];

                dtype = Dtype::from_string(hlo_primitive_type_to_string(elem_shape.element_type()));
                tensor_shape = get_tensor_shape_from_xla_shape(elem_shape);
                memory_layout = get_memory_layout_from_xla_shape(elem_shape);

                output_tensor_spec_list.push_back(TensorSpec(dtype, tensor_shape, memory_layout));
            }
        }

        std::shared_ptr<T> node = std::make_shared<T>(node_name, operands, output_tensor_spec_list);

        return node;
    }

    #define INSTANCIATE_TEMPLATE(op_name, op_code, op_class_code, ...) \
        template std::shared_ptr<op_class_code> IRParser::get_sketchy_node<op_class_code>( \
            std::shared_ptr<Graph> graph, \
            xla::HloInstruction *instruction \
        );

    OP_TYPE_LIST(INSTANCIATE_TEMPLATE)
    #undef INSTANCIATE_TEMPLATE


    std::shared_ptr<Op> IRParser::get_node_from_hlo_instruction(
        std::shared_ptr<Graph> graph,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {

        #define SWITCH_NODE(op_name, op_code, op_class_code, hlo_code) \
        case xla::HloOpcode::k##hlo_code: \
        { \
            std::shared_ptr<op_class_code> sketchy_node = get_sketchy_node<op_class_code>(graph, instruction); \
            return initialize_##op_code(sketchy_node, instruction, fused_computation); \
        }

        switch (instruction->opcode())
        {
            OP_TYPE_LIST(SWITCH_NODE)
        default:
            LOG(FATAL) << "Unsupported op " << instruction->name() << " op code " << instruction->opcode();  
            break;
        }

        #undef SWITCH_NODE
    }

    std::shared_ptr<Op> IRParser::initialize_abs(
        std::shared_ptr<Abs> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_add(
        std::shared_ptr<Add> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_bitcast(
        std::shared_ptr<Bitcast> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_broadcast(
        std::shared_ptr<Broadcast> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        std::vector<int> dimensions;

        for (auto const &d : instruction->dimensions()) {
            dimensions.push_back((int)d);
        }

        node->set_dimensions(dimensions);

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_clamp(
        std::shared_ptr<Clamp> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }

    std::shared_ptr<Op> IRParser::initialize_compare(
        std::shared_ptr<Compare> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        MathOp comparator;
        
        xla::ComparisonDirection direction = dynamic_cast<xla::HloCompareInstruction *>(instruction)->direction();
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

    std::shared_ptr<Op> IRParser::initialize_concatenate(
        std::shared_ptr<Concatenate> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        xla::HloConcatenateInstruction *concatenate_inst = reinterpret_cast<xla::HloConcatenateInstruction *>(instruction);
        node->set_dimension((int)concatenate_inst->concatenate_dimension());

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_constant(
        std::shared_ptr<Constant> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        if (node->get_output_spec(0).rank() == 0) {
            std::string value = instruction->literal().ToStringWithoutShape();
            node->set_value(value);
        }

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_convert(
        std::shared_ptr<Convert> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_convolution(
        std::shared_ptr<Convolution> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }

    std::shared_ptr<Op> IRParser::initialize_copy(
        std::shared_ptr<Copy> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_custom_call(
        std::shared_ptr<CustomCall> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        xla::HloCustomCallInstruction *custom_call = dynamic_cast<xla::HloCustomCallInstruction *>(instruction);
    
        node->set_custom_call_target(custom_call->custom_call_target());
        node->set_backend_config_str(custom_call->opaque());

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_divide(
        std::shared_ptr<Divide> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_exp(
        std::shared_ptr<Exp> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_gather(
        std::shared_ptr<Gather> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        xla::HloGatherInstruction *gather_inst = dynamic_cast<xla::HloGatherInstruction *>(instruction);
        int index_vector_dim;
        std::vector<int> offset_dims;
        std::vector<int> slice_sizes;
        std::vector<int> collapsed_slice_dims;
        std::vector<int> start_index_map;
        bool indices_are_sorted;
        bool unique_indices;

        const auto &gather_dimension_numbers = gather_inst->gather_dimension_numbers();
        
        index_vector_dim = (int)gather_dimension_numbers.index_vector_dim();

        for (auto const &d: gather_dimension_numbers.offset_dims()) {
            offset_dims.push_back((int)d);
        }

        for (auto const &d: gather_inst->gather_slice_sizes()) {
            slice_sizes.push_back((int)d);
        }


        for (auto const &d: gather_dimension_numbers.collapsed_slice_dims()) {
            collapsed_slice_dims.push_back((int)d);
        }

        for (auto const &d: gather_dimension_numbers.start_index_map()) {
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

    std::shared_ptr<Op> IRParser::initialize_get_tuple_element(
        std::shared_ptr<GetTupleElement> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }

    std::shared_ptr<Op> IRParser::initialize_iota(
        std::shared_ptr<Iota> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        int iota_dimension = (int)dynamic_cast<xla::HloIotaInstruction *>(instruction)->iota_dimension();

        node->set_iota_dimension(iota_dimension);

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_maximum(
        std::shared_ptr<Maximum> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_minimum(
        std::shared_ptr<Minimum> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_multiply(
        std::shared_ptr<Multiply> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_pad(
        std::shared_ptr<Pad> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        std::vector<int> padding_low;
        std::vector<int> padding_high;

        xla::HloPadInstruction *padding = dynamic_cast<xla::HloPadInstruction *>(instruction);

        Scalar padding_value = hlo_constant_to_scalar(dynamic_cast<xla::HloConstantInstruction const *>(padding->padding_value()));

        for (auto const &config : padding->padding_config().dimensions()) {
            EXPECT_TRUE(config.interior_padding() == 0, "Interior padding not supported"); 
            EXPECT_TRUE(config.edge_padding_high() >= 0, "Padding must non-negative");
            EXPECT_TRUE(config.edge_padding_low() >= 0, "Padding must non-negative");

            padding_low.push_back((int)config.edge_padding_low());
            padding_high.push_back((int)config.edge_padding_high());
        }

        node->set_padding_low(padding_low);
        node->set_padding_high(padding_high);

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_parameter(
        std::shared_ptr<Parameter> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        xla::HloParameterInstruction *parameter = static_cast<xla::HloParameterInstruction *>(instruction);
        node->set_parameter_number((int)parameter->parameter_number());
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_reduce(
        std::shared_ptr<Reduce> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        EXPECT_TRUE(instruction->operand_count() == 2, "");
        node->set_dimension((int)instruction->dimensions()[0]);

        xla::HloInstruction *constant = dynamic_cast<xla::HloReduceInstruction *>(instruction)->init_values()[0];
        
        Scalar init_value = hlo_constant_to_scalar(dynamic_cast<xla::HloConstantInstruction *>(constant));

        Reducer reducer;

        if (instruction->to_apply()->root_instruction()->opcode() == xla::HloOpcode::kAdd) {
            reducer = Reducer::Sum;
        } else if (instruction->to_apply()->root_instruction()->opcode() == xla::HloOpcode::kMaximum) {
            reducer = Reducer::Max;
        } else if (instruction->to_apply()->root_instruction()->opcode() == xla::HloOpcode::kMinimum) {
            reducer = Reducer::Min;
        }  else {
            LOG(FATAL) << "Unsupported reduction operation " << instruction->to_apply()->root_instruction()->name() << std::endl;
        }

        node->set_reducer(reducer);

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_reduce_window(
        std::shared_ptr<ReduceWindow> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }

    std::shared_ptr<Op> IRParser::initialize_reshape(
        std::shared_ptr<Reshape> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_rsqrt(
        std::shared_ptr<Rsqrt> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_select(
        std::shared_ptr<Select> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_slice(
        std::shared_ptr<Slice> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        xla::HloSliceInstruction *slice_inst = static_cast<xla::HloSliceInstruction *>(instruction);

        std::vector<int> slice_starts;
        std::vector<int> slice_limits;
        std::vector<int> slice_strides;

        int rank = node->get_output_spec(0).get_shape().rank();

        for (int idx = 0; idx < rank; ++idx) {
            slice_starts.push_back((int)slice_inst->slice_starts(idx));
            slice_limits.push_back((int)slice_inst->slice_limits(idx));
            slice_strides.push_back((int)slice_inst->slice_strides(idx));
        }

        node->set_slice_starts(slice_starts);
        node->set_slice_limits(slice_limits);
        node->set_slice_strides(slice_strides);

        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_subtract(
        std::shared_ptr<Subtract> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_tanh(
        std::shared_ptr<Tanh> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        return std::static_pointer_cast<Op>(node);
    }

    std::shared_ptr<Op> IRParser::initialize_transpose(
        std::shared_ptr<Transpose> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }

    std::shared_ptr<Op> IRParser::initialize_tuple(
        std::shared_ptr<Tuple> node,
        xla::HloInstruction *instruction, 
        std::unordered_map<std::string, xla::HloComputation *> *fused_computation
    ) {
        LOG(FATAL) << "Unimplemented";
    }
}
}