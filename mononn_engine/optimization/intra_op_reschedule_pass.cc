#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/intra_op_reschedule_pass.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_annotation/cluster_type.h"
#include "mononn_engine/core/op/cluster_reduce.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"

namespace mononn_engine {
namespace optimization {
    using ClusterType = mononn_engine::core::op_annotation::ClusterType;
    using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
    using TensorShape = mononn_engine::core::tensor::TensorShape;
    using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
    using ClusterReduce = mononn_engine::core::op::ClusterReduce;

    std::string IntraOpReschedulePass::name() const {
        return PassName::IntraOpReschedulePass;
    }

    bool IntraOpReschedulePass::run(Graph *graph, std::shared_ptr<CUDAContext> cuda_context) {
        if (this->ilp_factor == 1) return true;

        ClusterOp *cluster_node = graph->get_node(this->cluster_node_name)->as<ClusterOp>();

        if (cluster_node->has_attribute(OpAttribute::intra_op_reschedule_factor)) {
            LOG(FATAL) << "Node " << cluster_node->get_name() << " have already been optimized by " << this->name() <<
                    ". Value: " << cluster_node->get_attribute(OpAttribute::intra_op_reschedule_factor);
        }

        LOG(INFO) << this->name() << " for " << cluster_node->get_name() << " with factor " << this->ilp_factor;

        if (cluster_node->get_cluster_type() == ClusterType::Elewise) {
            return this->run_for_elewise_cluster(graph, cuda_context, cluster_node);
        } else if (cluster_node->get_cluster_type() == ClusterType::Reduce) {
            return this->run_for_reduce_cluster(graph, cuda_context, cluster_node);
        } else {
            LOG(FATAL) << "Unsupported cluster type: " << cluster_node->get_cluster_type().to_string();
        }
    }

    bool IntraOpReschedulePass::run_for_elewise_cluster(
            Graph *graph,
            std::shared_ptr<CUDAContext> cuda_context,
            ClusterOp *cluster_node) {

        TensorShape loop_shape = cluster_node->get_schedule().get_loop_shape(0);

        if (loop_shape.element_count() <
            this->ilp_factor * cuda_context->cuda_runtime_context.grid_dim.XYZ() * cuda_context->cuda_runtime_context.block_dim.XYZ()) {
            LOG(FATAL) << cluster_node->get_name() << " Cannot be instruction paralleled with factor " << this->ilp_factor << "\n" <<
            "Loop shape: " << loop_shape.to_string() << "\n" <<
            "CUDA grid dim " << cuda_context->cuda_runtime_context.grid_dim.to_string() << "\n" <<
            "CUDA block dim " << cuda_context->cuda_runtime_context.block_dim.to_string() << "\n";
        }

        cluster_node->set_instruction_parallel_factor(this->ilp_factor);

        return true;
    }

    bool IntraOpReschedulePass::run_for_reduce_cluster(
            Graph *graph,
            std::shared_ptr<CUDAContext> cuda_context,
            ClusterOp *cluster_node) {

        cluster_node->set_instruction_parallel_factor(this->ilp_factor);

        return true;
    }

    bool IntraOpReschedulePass::can_rescheduled_with_ilp_factor(
            std::shared_ptr<CUDAContext> cuda_context,
            ClusterOp *cluster_node,
            int ilp_factor) {
        if (ilp_factor == 1) return true;

        if (cluster_node->is_cluster_elewise()) {
            int element_count = cluster_node->get_schedule().get_loop_shape(0).element_count();
            return element_count >=
                cuda_context->cuda_runtime_context.grid_dim.XYZ() *
                cuda_context->cuda_runtime_context.block_dim.XYZ() *
                ilp_factor;
        } else if (cluster_node->is_cluster_reduce()) {
            LocalityTier::Tier tier = cluster_node->get_schedule().get_locality_tier();
            int reduction_dimension_size = cluster_node->as<ClusterReduce>()->get_reduction_dimension_size();

            if (tier == LocalityTier::kT1) {
                return reduction_dimension_size >= (cuda_context->cuda_device_context.warp_size * ilp_factor);
            } else if (tier == LocalityTier::kT2) {
                return reduction_dimension_size >= (cuda_context->cuda_runtime_context.block_dim.XYZ() * ilp_factor);
            } else {
                LOG(FATAL) << "";
            }
        } else {
            LOG(FATAL) << "";
        }
    }
}
}