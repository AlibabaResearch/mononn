#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/global_synchronization_assignment_pass.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/gpu/synchronization.h"

namespace mononn_engine {
namespace optimization {
    using OpType = mononn_engine::core::op::OpType;
    using Synchronization = mononn_engine::core::gpu::Synchronization;

    std::string GlobalSynchronizationAssignmentPass::name() const {
        return PassName::GlobalSynchronizationAssignmentPass;
    }

    bool GlobalSynchronizationAssignmentPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        for (auto const &node_name : graph->get_node_list()) {
            for (auto &edge : graph->get_node_output_edges(node_name)) {
                if (edge->get_dst()->get_type() == OpType::get_tuple_element) {
                    edge->set_sync(Synchronization::None);
                    continue;
                }

                if (edge->get_src()->get_type() == OpType::parameter ||
                    edge->get_src()->get_type() == OpType::constant ||
                    edge->get_src()->get_type() == OpType::iota) {
                    edge->set_sync(Synchronization::None);
                } else {
                    edge->set_sync(Synchronization::Global);
                }
            }
        }

        return true;
    }
}
}
