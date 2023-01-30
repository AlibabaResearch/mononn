#include "mononn_engine/optimization/regional_synchronization_assignment_pass.h"

#include "mononn_engine/core/gpu/synchronization.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpType = mononn_engine::core::op::OpType;
using Op = mononn_engine::core::op::Op;
using Synchronization = mononn_engine::core::gpu::Synchronization;
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;

std::string RegionalSynchronizationAssignmentPass::name() const {
  return PassName::RegionalSynchronizationAssignmentPass;
}

bool RegionalSynchronizationAssignmentPass::run(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    std::shared_ptr<ClusterOp> cluster_node =
        std::static_pointer_cast<ClusterOp>(graph->get_node(cluster_node_name));
    if (!cluster_node->is_cluster_reduce()) {
      continue;
    }

    for (auto const& node_name : cluster_node->get_graph()->get_node_list()) {
      std::shared_ptr<Op> node = cluster_node->get_graph()->get_node(node_name);
      if (node->get_type() != OpType::reduce) {
        continue;
      }

      for (auto& edge :
           cluster_node->get_graph()->get_node_output_edges(node_name)) {
        if (edge->get_dst()->has_attribute(
                OpAttribute::on_chip_transfer_from_node) &&
            edge->get_dst()->get_attribute(
                OpAttribute::on_chip_transfer_from_node) == node_name) {
          LocalityTier::Tier tier =
              cluster_node->get_schedule().get_locality_tier();

          if (tier == LocalityTier::kT1)
            edge->set_sync(Synchronization::Warp);
          else if (tier == LocalityTier::kT2)
            edge->set_sync(Synchronization::ThreadBlock);
          else
            LOG(FATAL) << "Unsupported locality tier: " << tier.to_string();
        }
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine