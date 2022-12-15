#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/cache_bypass_pass.h"


namespace mononn_engine {
namespace optimization {
    std::string CacheBypassPass::name() const {
        return PassName::CacheBypassPass;
    }

    bool CacheBypassPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        
    }
}
}