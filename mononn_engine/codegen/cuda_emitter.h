#pragma once 

#include <sstream>
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/context/index_trace_stamp.h"
#include "mononn_engine/core/graph/graph.h"

namespace mononn_engine {
namespace codegen {
    using Op = mononn_engine::core::op::Op;
    using Graph = mononn_engine::core::graph::Graph;
    using ConcreteIndexStamp = mononn_engine::core::context::ConcreteIndexStamp;
    using SymbolicIndexStamp = mononn_engine::core::context::SymbolicIndexStamp;

    class CodegenState;
    class CodegenStateSpace;


    class CUDAEmitter {
    public:
        CUDAEmitter(const Graph *_graph, const CodegenStateSpace *_codegens_state_space) 
            : graph(_graph), codegens_state_space(_codegens_state_space) {}

        static bool should_use_new_code_emitter(const Graph *graph);
        static std::string emit_code_use_new_code_emitter(
            const Graph *graph, 
            const std::string &linear_loop_key, 
            const std::string &linear_loop_stride, 
            const std::string &strided_loop_key = "");

        void emit(const Op *node, const ConcreteIndexStamp &concrete_index);
        // void emit(const Op *node, const std::map<std::string, std::string> &symbolic_index_initializer);
        void emit_codegen_state(const Op *node, const CodegenState &codegen_state);

        void emit_constant(const Op *node, const CodegenState &codegen_state);
        void emit_elementwise_unary(const Op *node, const CodegenState &codegen_state);
        void emit_elementwise_binary(const Op *node, const CodegenState &codegen_state);
        // void emit_gather(const Op *node, const CodegenState &codegen_state);
        void emit_parameter(const Op *node, const CodegenState &codegen_state);
        void emit_reduce(const Op *node, const CodegenState &codegen_state);
        void emit_reduce_window(const Op *node, const CodegenState &codegen_state);
        void emit_select(const Op *node, const CodegenState &codegen_state);
        void emit_non_op(const Op *node, const CodegenState &codegen_state);
        void emit_output(const Op *node, const CodegenState &codegen_state);
        void emit_output(const Op *node, const ConcreteIndexStamp &concrete_index);

        void emit_loop_end(const Op *node);
        void emit_post_reduce_if(const Op *node);
        void emit_post_reduce_if_end(const Op *node);
        void emit_reduce_broadcast(const Op *node);

        const std::stringstream &get_code_stream() const;
    private:
        const CodegenStateSpace *codegens_state_space;
        std::stringstream code_stream;
        const Graph *graph;
    };

    // Compbination state:
    // Reduce window
    // ILP (instantiate index)
    // Traced index
    struct CodegenState {
        std::string node_name_suffix;
        ConcreteIndexStamp index;
    };

    class CodegenStateSampler {
    public:
        CodegenStateSampler(std::function<CodegenState(const CodegenState&)> _sample_func) : sample_func(_sample_func) {}

        CodegenState sample(const CodegenState &codegen_state) const;

    private:
        std::function<CodegenState(const CodegenState&)> sample_func;
    };
    
    class CodegenStateSpace {
    public:
        using StateSamplerArray = std::vector<CodegenStateSampler>;

        void push(const StateSamplerArray &state_array);
        void pop();

        std::vector<CodegenState> generate_codegen_state(const CodegenState &init_state) const;
        std::vector<CodegenState> generate_codegen_state(const std::vector<CodegenState> &init_state_list) const;

    private:
        std::vector<StateSamplerArray> space_array;
    };

    class CodegenStateSpaceMgr {
    public:
        CodegenStateSpaceMgr();

        void emit_instruction_level_parallelism(int ilp_factor);

        void emit_op(const Op *node);
        void emit_reduce_window(const Op *node);
        void emit_slice(const Op *node);

        void recall_op(const Op *node);
        void recall_reduce_window(const Op *node);
        void recall_slice(const Op *node);

        const CodegenStateSpace *get_codegen_state_space() const;
    private:
        std::unique_ptr<CodegenStateSpace> codegen_state_space;
    };
}
}