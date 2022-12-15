#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/cache_temporal_access_pass.h"

namespace mononn_engine {
namespace optimization {
    std::string CacheTemporalAccessPass::name() const {
        return PassName::CacheTemporalAccessPass;
    }

    bool CacheTemporalAccessPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        
    }
}
}