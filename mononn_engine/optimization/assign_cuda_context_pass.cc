#include "mononn_engine/optimization/common.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/optimization/assign_cuda_context_pass.h"

namespace mononn_engine {
namespace optimization {
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using OpType = mononn_engine::core::op::OpType;
    std::string AssignCUDAContextPass::name() const {
        return PassName::AssignCUDAContextPass;
    }

    bool AssignCUDAContextPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        for (auto const &cluster_node_name : graph->get_node_list_by_type(OpType::cluster)) {
            auto cluster_node = graph->get_node(cluster_node_name)->as<ClusterOp>();
            cluster_node->set_cuda_context(cuda_context);
        }

        return true;
    }
}
}