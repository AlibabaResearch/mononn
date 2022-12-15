#pragma once 
#include <string>
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using Graph = mononn_engine::core::graph::Graph;
    
    class MemoryIntensiveOnChipFusionPass : public GraphPass {
    public:
        std::string name() const override;
        bool run(Graph *graph) override;

    private:

    };
}
}