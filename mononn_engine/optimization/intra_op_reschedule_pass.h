#pragma once

#include "mononn_engine/optimization/graph_pass.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/helpers/helpers.h"

namespace mononn_engine {
namespace optimization {
    using ClusterOp = mononn_engine::core::op::ClusterOp;

    class IntraOpReschedulePass : public GraphPass{
    public:
        IntraOpReschedulePass(std::string _cluster_node_name, int _ilp_factor) : cluster_node_name(_cluster_node_name), ilp_factor(_ilp_factor) {}

        std::string name() const override;
        bool run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) override;

        static bool can_rescheduled_with_ilp_factor(
            std::shared_ptr<CUDAContext> cuda_context,
            ClusterOp *cluster_node,
            int ilp_factor);
    private:

        bool run_for_elewise_cluster(Graph *graph, std::shared_ptr<CUDAContext> cuda_context, ClusterOp *cluster_node);
        bool run_for_reduce_cluster(Graph *graph, std::shared_ptr<CUDAContext> cuda_context, ClusterOp *cluster_node);

        std::string cluster_node_name;
        int ilp_factor;
    };
}
}


