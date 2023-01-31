// Copyright 2023 The MonoNN Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mononn_engine/optimization/locality_escalation_pass.h"

#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/auxiliary_impl_type.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_annotation/op_attribute.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/op_impl/output_reg_impl.h"
#include "mononn_engine/core/op_impl/parameter_read_reg_impl.h"
#include "mononn_engine/core/op_impl/parameter_shfl_impl.h"
#include "mononn_engine/core/op_impl/parameter_smem_impl.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/helpers/helpers.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using OpAttribute = mononn_engine::core::op_annotation::OpAttribute;
using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using ParameterShflImpl = mononn_engine::core::op_impl::ParameterShflImpl;
using ParameterSmemIMpl = mononn_engine::core::op_impl::ParameterSmemImpl;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using Tensor = mononn_engine::core::tensor::Tensor;
using ParameterReadRegImpl = mononn_engine::core::op_impl::ParameterReadRegImpl;
using OutputRegImpl = mononn_engine::core::op_impl::OutputRegImpl;
using AuxiliaryImplType = mononn_engine::core::op_annotation::AuxiliaryImplType;

std::string LocalityEscalationPass::name() const {
  return PassName::LocalityEscalationPass;
}

bool LocalityEscalationPass::run(Graph* graph,
                                 std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);
    for (auto const& node_name :
         cluster_node->as<ClusterOp>()->get_graph()->get_node_list()) {
      std::shared_ptr<Op> node =
          cluster_node->as<ClusterOp>()->get_graph()->get_node(node_name);
      if (!node->has_attribute(OpAttribute::on_chip_transfer_from_node)) {
        continue;
      }
      std::string preceding_node_name =
          node->get_attribute(OpAttribute::on_chip_transfer_from_node);
      std::shared_ptr<Op> preceding_node =
          cluster_node->as<ClusterOp>()->get_graph()->get_node(
              preceding_node_name);

      LOG(INFO) << "In cluster " << cluster_node_name
                << " escalate locality for node " << preceding_node_name
                << " and " << node_name << ".";

      if (preceding_node->get_type() == OpType::reduce) {
        LocalityTier::Tier tier =
            cluster_node->as<ClusterOp>()->get_schedule().get_locality_tier();
        if (tier == LocalityTier::kT1) {
          ParameterShflImpl::InputSpec input_spec;
          input_spec.operand =
              Tensor(preceding_node_name, preceding_node->get_output_spec(0));
          Tensor output(node_name, node->get_output_spec(0));
          std::shared_ptr<OpImplBase> impl =
              ParameterShflImpl::get_available_implementations(
                  cuda_context, input_spec, output)[0];
          impl->set_hlo_text(node->get_hlo_text());
          node->set_implementation(impl);
          // node->propagate_index_to_implementation();
        } else if (tier == LocalityTier::kT2) {
          ParameterSmemIMpl::InputSpec input_spec;
          input_spec.operand =
              Tensor(preceding_node_name, preceding_node->get_output_spec(0));
          Tensor output(node_name, node->get_output_spec(0));
          std::shared_ptr<OpImplBase> impl =
              ParameterSmemIMpl::get_available_implementations(
                  cuda_context, input_spec, output)[0];
          impl->set_hlo_text(node->get_hlo_text());
          node->set_implementation(impl);
          // node->propagate_index_to_implementation();
        } else {
          LOG(FATAL) << "Unsupported locality tier for reduce: "
                     << tier.to_string();
        }

        cluster_node->as<ClusterOp>()->get_graph()->add_edge(
            preceding_node_name, node_name);
      } else {
        std::string reg_buffer_name = preceding_node_name + "_reg_buffer";
        std::string step_id = cluster_node->as<ClusterOp>()
                                  ->get_schedule()
                                  .get_inner_loop()
                                  .get_loop_step_id();
        std::string step_cnt = cluster_node->as<ClusterOp>()
                                   ->get_schedule()
                                   .get_inner_loop()
                                   .get_loop_steps();
        Tensor preceding_node_tensor(preceding_node_name,
                                     preceding_node->get_output_spec(0));

        OutputRegImpl::InputSpec output_reg_impl_spec;
        output_reg_impl_spec.operand = preceding_node_tensor;
        output_reg_impl_spec.reg_buffer_name = reg_buffer_name;
        output_reg_impl_spec.step_id = step_id;
        output_reg_impl_spec.step_cnt = step_cnt;

        ParameterReadRegImpl::InputSpec parameter_read_reg_impl_spec;
        parameter_read_reg_impl_spec.operand_reg_buffer_name = reg_buffer_name;
        parameter_read_reg_impl_spec.step_id = step_id;

        Tensor node_tensor(node_name, node->get_output_spec(0));

        std::shared_ptr<OutputRegImpl> output_reg_impl =
            std::make_shared<OutputRegImpl>(cuda_context, output_reg_impl_spec);
        std::shared_ptr<ParameterReadRegImpl> parameter_read_reg_impl =
            std::make_shared<ParameterReadRegImpl>(
                cuda_context, parameter_read_reg_impl_spec, node_tensor);

        output_reg_impl->set_hlo_text("//Write to reg buffer");
        parameter_read_reg_impl->set_hlo_text(node->get_hlo_text());

        preceding_node->add_auxiliary_impl(
            AuxiliaryImplType::buffer_in_register,
            std::static_pointer_cast<OpImplBase>(output_reg_impl));
        node->set_implementation(
            std::static_pointer_cast<OpImplBase>(parameter_read_reg_impl));
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine