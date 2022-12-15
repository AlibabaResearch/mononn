#pragma onece 

#include <string>

#include "mononn_engine/core/graph/clustered_graph.h"
#include "mononn_engine/codegen/cuda_program.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace codegen {
    class ClusteredGraphCodegen {
    public:
        using CUDAProgram = mononn_engine::codegen::CUDAProgram;
        using CUDAContext = mononn_engine::core::context::CUDAContext;
        using ClusteredGraph = mononn_engine::core::graph::ClusteredGraph;

        static CUDAProgram generate(std::shared_ptr<CUDAContext> cuda_context, std::shared_ptr<ClusteredGraph> graph);
    private:

        static void initialize_buffer_manager(std::shared_ptr<ClusteredGraph> graph);
        static void synchronization_analysis(std::shared_ptr<ClusteredGraph> graph);
    };
}
}