#pragma once
#include "mononn_engine/optimization/graph_pass.h"

namespace mononn_engine {
namespace optimization {
    using Op = mononn_engine::core::op::Op;

    class MergeIndependentPass : public GraphPass {
    public:
        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;
    private:

        static std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> merge_node_pairs_by_graph;

        void do_merge(std::shared_ptr<Op> &node1, std::shared_ptr<Op> &node2, Graph *graph);
    };

}
}
