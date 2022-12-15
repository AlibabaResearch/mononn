#include "mononn_engine/optimization/transpose_in_smem_pass.h"

namespace mononn_engine {
namespace optimization {
    std::string TransposeInSmemPass::name() const {
        return "TransposeInSmemPass";
    }

    bool TransposeInSmemPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {}
}
}