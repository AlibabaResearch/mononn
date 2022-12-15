#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/schedule_assignment_pass.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/core/op/cluster_op.h"

namespace mononn_engine {
namespace optimization {
    using Op = mononn_engine::core::op::Op;
    using ClusterOp = mononn_engine::core::op::ClusterOp;
    using OpType = mononn_engine::core::op::OpType;
    using Schedule = mononn_engine::core::schedule::Schedule;
    using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;

    std::string ScheduleAssignmentPass::name() const {
        return PassName::ScheduleAssignmentPass;
    }

    bool ScheduleAssignmentPass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        for (auto const &graph_node_name : graph->get_node_list()) {
            std::shared_ptr<Op> graph_node = graph->get_node(graph_node_name);
            if (graph_node->get_type() == OpType::cluster) {
                if (graph_node->is_cluster_elewise()) {
                    Schedule schedule = graph_node->as<ClusterOp>()->construct_schedule(LocalityTier::kT0);
                    graph_node->as<ClusterOp>()->set_schedule(schedule);
                } else if (graph_node->is_cluster_reduce()) {
                    LocalityTier::Tier reduce_tier = this->graph_specification->cluster_reduce_spec().at(graph_node_name).locality_tier();
                    Schedule schedule = graph_node->as<ClusterOp>()->construct_schedule(reduce_tier);
                    graph_node->as<ClusterOp>()->set_schedule(schedule);
                } else {
                    LOG(FATAL) << "Unsupported cluster type" << graph_node->as<ClusterOp>()->get_cluster_type().to_string();
                }
            }
        }

        return true;
    }
}
}