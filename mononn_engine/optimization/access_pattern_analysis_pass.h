#pragma once

#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
    class AccessPatternAnalysisPass : public GraphPass {
    public:
        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;
    private:
        // MonoNN typically use as much Smem as need.
        // Thus A100 will left 192KB - 164KB = 28KB L1
        // A10 will left 128KB - 100KB = 28 KB L1
        // Thus 20KB L1 is a reasonable value for temporal access buffer
        const int l1_cache_limit_in_bytes = 10 * 1024;
    };
}
}

