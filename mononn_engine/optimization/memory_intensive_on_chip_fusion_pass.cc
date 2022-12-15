#include <unordered_set>

#include "mononn_engine/optimization/memory_intensive_on_chip_fusion_pass.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"

namespace mononn_engine {
namespace optimization {
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;

    std::string MemoryIntensiveOnChipFusionPass::name() const {
        return "MemoryIntensiveOnChipFusionPass";
    }

    bool MemoryIntensiveOnChipFusionPass::run(Graph *graph) {
        std::unordered_set<std::string> visit;

        return true;
    }   
}
}