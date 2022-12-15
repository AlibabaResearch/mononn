#pragma once
#include "mononn_engine/optimization/graph_pass.h"
namespace mononn_engine {
namespace optimization {
    // Only memory read identified both *streaming* and *vectorized* will be async prefetched into smem.
    // Streaming only memory read will use ld with cache by pass policy.
    // Nodes that introduce dynamic index such as gather, scatter cannot be prefetched.
    // Pad is also not allowed in the prefetched graph as this operator will introduce difficulity in infer buffer size.
    class SmemPrefetchPass : public GraphPass {
    public:
        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;
    private:

    };
} // onefuser
} // optimization
