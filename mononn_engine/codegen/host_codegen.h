#pragma once

#include <memory>

#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/graph/graph.h"

namespace mononn_engine {
namespace codegen {
    class HostCodegen {
    public:
        using Op = mononn_engine::core::op::Op;
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using Graph = mononn_engine::core::graph::Graph;

        static std::string generate(std::shared_ptr<CUDAContext> cuda_context, Graph *graph, const std::string &kernel_name);
    private:

        static std::string generate_stream(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_memory_allocation(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_memory_initialization(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_parameter_declaration(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_parameter_initialization(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_kernel_invocation(std::shared_ptr<CUDAContext> cuda_context, Graph *graph, const std::string &kernel_name);
        static std::string generate_print_output(std::shared_ptr<CUDAContext> cuda_context, Graph *graph);
        static std::string generate_stream_synchronize(std::string stream);
    };
}
}