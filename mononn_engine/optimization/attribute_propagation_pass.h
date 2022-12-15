#pragma once

#include "mononn_engine/optimization/graph_pass.h"
#include "mononn_engine/core/context/cuda_context.h"

namespace mononn_engine {
namespace optimization {

    class AttributePropagationPass : public GraphPass {
    public:
        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;
    };

} // onefuser
} // optimization
