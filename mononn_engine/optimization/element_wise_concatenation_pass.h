#pragma once

#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
    // concatenate elementwise clusters
    class ElementWiseConcatenationPass : public GraphPass {
    public:

        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;
    private:

    };
}
}
