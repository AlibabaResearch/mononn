#include "mononn_engine/optimization/common.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/optimization/trace_symbolic_index_pass.h"

namespace mononn_engine {
namespace optimization {
    using OpType = mononn_engine::core::op::OpType;
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;

    std::string TraceSymbolicIndexPass::name() const {
        return PassName::TraceSymbolicIndexPass;
    }

    bool TraceSymbolicIndexPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        for (auto const &cluster_node_name : graph->get_node_list_by_type(OpType::cluster)) {
            auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();

            if (cluster_node->get_cluster_type() == ClusterType::Elewise ||
                cluster_node->get_cluster_type() == ClusterType::Reduce) {

                cluster_node->trace_symbolic_index();
            }
        }

        return true;
    }
}
}