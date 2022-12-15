#pragma once

#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <memory>

#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/edge/edge.h"
#include "mononn_engine/core/op/all_operators.h"
#include "mononn_engine/core/op/all_cluster_operators.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace mononn_engine {
namespace parser {
    // Parser hlo ir according hlo fusoin
    class IRParserFused {
    public:
        using Op = mononn_engine::core::op::Op;
        using TensorSpec = mononn_engine::core::tensor::TensorSpec;
        using Graph = mononn_engine::core::graph::Graph;
        using Scalar = mononn_engine::core::tensor::Scalar;

        static std::shared_ptr<Graph> from_file(std::string file_name);
        static std::shared_ptr<Graph> from_text(std::string text);
        static std::shared_ptr<Graph> from_hlo_module_proto_file(std::string path_to_proto);
        static std::shared_ptr<Graph> from_hlo_module(const xla::HloModule *hlo_module);
        static std::unique_ptr<Graph> from_hlo_module_unique(const xla::HloModule *hlo_module);

    private:
        static std::shared_ptr<Op> get_node_from_hlo_instruction(
            Graph *graph,
            xla::HloInstruction *instruction, 
            std::unordered_map<std::string, xla::HloComputation *> *fused_computation
        );

        static void from_hlo_module_impl(Graph *graph, const xla::HloModule *hlo_module);

        #define JOIN_USING(x, y) x::y

        #define USING_OP(op_name, op_code, op_class_code, ...) \
        using op_class_code = JOIN_USING(mononn_engine::core::op, op_class_code);

        OP_TYPE_LIST(USING_OP)
        OP_TYPE_LIST_CLUSTER(USING_OP)

        #undef USING_OP
        #undef JOIN_USING
        
        #define DECLARE_OP_PARSER(op_name, op_code, op_class_code, ...) \
            static std::shared_ptr<Op> initialize_##op_code( \
                std::shared_ptr<op_class_code> node, \
                xla::HloInstruction *instruction, \
                std::unordered_map<std::string, xla::HloComputation *> *fused_computation \
            );

        OP_TYPE_LIST(DECLARE_OP_PARSER)
        OP_TYPE_LIST_CLUSTER(DECLARE_OP_PARSER)
        #undef DECLARE_OP_PARSER


        template<typename T>
        static std::shared_ptr<T> get_sketchy_node(
            Graph *graph,
            xla::HloInstruction *instruction
        );

        static Scalar hlo_constant_to_scalar(xla::HloConstantInstruction const *constant);

        // static std::string to_variable_name(std::string name);

        static std::vector<std::shared_ptr<Op>> get_operands(
                Graph *graph,
                xla::HloInstruction *instruction);

        static std::vector<TensorSpec> get_output_spec_list(
                xla::HloInstruction *instruction
        );
    };
}
}