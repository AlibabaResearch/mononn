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

#include "mononn_engine/optimization/implementation_assignment_pass.h"

#include "mononn_engine/core/gpu/cutlass/conv_backend_config.h"
#include "mononn_engine/core/gpu/cutlass/cutlass_config.h"
#include "mononn_engine/core/gpu/cutlass/gemm_backend_config.h"
#include "mononn_engine/core/op/cluster_op.h"
#include "mononn_engine/core/op/custom_call.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/core/op/op_type.h"
#include "mononn_engine/core/op_annotation/locality_tier.h"
#include "mononn_engine/core/op_impl/conv_impl.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/core/op_impl/op_impl_base.h"
#include "mononn_engine/core/tensor/tensor.h"
#include "mononn_engine/helpers/env_variable.h"
#include "mononn_engine/optimization/common.h"

namespace mononn_engine {
namespace optimization {
using Op = mononn_engine::core::op::Op;
using Tensor = mononn_engine::core::tensor::Tensor;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
using ConvImpl = mononn_engine::core::op_impl::ConvImpl;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using CutlassConfig = mononn_engine::core::gpu::cutlass::CutlassConfig;
using GemmBackendConfig = mononn_engine::core::gpu::cutlass::GemmBackendConfig;
using ConvBackendConfig = mononn_engine::core::gpu::cutlass::ConvBackendConfig;
using CustomCall = mononn_engine::core::op::CustomCall;
using EnvVar = mononn_engine::helpers::EnvVar;

std::string ImplementationAssignmentPass::name() const {
  return PassName::ImplementationAssignmentPass;
}

bool ImplementationAssignmentPass::run(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  for (auto const& graph_node_name : graph->get_node_list()) {
    std::shared_ptr<Op> graph_node = graph->get_node(graph_node_name);
    if (graph_node->get_type() == OpType::cluster) {
      if (graph_node->is_cluster_elewise()) {
        for (auto const node_name :
             std::static_pointer_cast<ClusterOp>(graph_node)
                 ->get_graph()
                 ->get_node_list()) {
          std::shared_ptr<Op> node =
              std::static_pointer_cast<ClusterOp>(graph_node)
                  ->get_graph()
                  ->get_node(node_name);
          std::vector<std::shared_ptr<OpImplBase>> impl_list =
              node->generate_candidate_implementation(cuda_context,
                                                      LocalityTier::kT0);
          node->set_implementation(impl_list[0]);
        }

      } else if (graph_node->is_cluster_reduce()) {
        LocalityTier::Tier reduce_tier =
            this->graph_specification->cluster_reduce_spec()
                .at(graph_node_name)
                .locality_tier();
        int reduce_implementation_id =
            this->graph_specification->cluster_reduce_spec()
                .at(graph_node_name)
                .reduce_implementation();
        for (auto const node_name :
             std::static_pointer_cast<ClusterOp>(graph_node)
                 ->get_graph()
                 ->get_node_list()) {
          std::shared_ptr<Op> node =
              std::static_pointer_cast<ClusterOp>(graph_node)
                  ->get_graph()
                  ->get_node(node_name);
          if (node->get_type() == OpType::reduce) {
            std::vector<std::shared_ptr<OpImplBase>> impl_list =
                node->generate_candidate_implementation(cuda_context,
                                                        reduce_tier);
            node->set_implementation(impl_list[reduce_implementation_id]);
          } else {
            std::vector<std::shared_ptr<OpImplBase>> impl_list =
                node->generate_candidate_implementation(cuda_context,
                                                        reduce_tier);
            node->set_implementation(impl_list[0]);
          }
        }

      } else {
        LOG(FATAL)
            << "Unsupported cluster type"
            << graph_node->as<ClusterOp>()->get_cluster_type().to_string();
      }
    } else if (graph_node->get_type() == OpType::custom_call) {
      if (graph_node->is_gemm()) {
        std::unique_ptr<CutlassConfig> cutlass_config =
            std::make_unique<CutlassConfig>();
        std::unique_ptr<GemmBackendConfig> gemm_backend_config =
            std::make_unique<GemmBackendConfig>();
        auto const& gemm_spec =
            this->graph_specification->gemm_spec_list().at(graph_node_name);
        cutlass_config->ParseFromProto(&gemm_spec.cutlass_config());
        gemm_backend_config->ParseFromProto(&gemm_spec.gemm_backend_config());

        GemmImpl::InputSpec input_spec;
        input_spec.A = Tensor(graph_node->get_operand(0)->get_name(),
                              graph_node->get_operand(0)->get_output_spec(0));
        input_spec.B = Tensor(graph_node->get_operand(1)->get_name(),
                              graph_node->get_operand(1)->get_output_spec(0));
        if (graph_node->get_operands().size() > 2) {
          input_spec.C = std::make_shared<Tensor>(
              graph_node->get_operand(2)->get_name(),
              graph_node->get_operand(2)->get_output_spec(0));
        }

        Tensor output(graph_node->get_name(), graph_node->get_output_spec(0));

        std::shared_ptr<GemmImpl> gemm_impl = std::make_shared<GemmImpl>(
            cuda_context, input_spec, *cutlass_config, *gemm_backend_config,
            output);

        graph_node->set_implementation(gemm_impl);
      } else if (graph_node->is_conv()) {
        std::unique_ptr<CutlassConfig> cutlass_config =
            std::make_unique<CutlassConfig>();
        std::unique_ptr<ConvBackendConfig> conv_backend_config =
            std::make_unique<ConvBackendConfig>();
        auto const& conv_spec =
            this->graph_specification->conv_spec_list().at(graph_node_name);
        cutlass_config->ParseFromProto(&conv_spec.cutlass_config());
        conv_backend_config->ParseFromProto(&conv_spec.conv_backend_config());

        ConvImpl::InputSpec input_spec;
        input_spec.A = Tensor(graph_node->get_operand(0)->get_name(),
                              graph_node->get_operand(0)->get_output_spec(0));
        input_spec.B = Tensor(graph_node->get_operand(1)->get_name(),
                              graph_node->get_operand(1)->get_output_spec(0));

        if (graph_node->get_operands().size() > 2) {
          input_spec.C = std::make_shared<Tensor>(
              graph_node->get_operand(2)->get_name(),
              graph_node->get_operand(2)->get_output_spec(0));
        }

        input_spec.filter_size =
            graph_node->as<CustomCall>()->get_filter_size();
        input_spec.filter_stride =
            graph_node->as<CustomCall>()->get_filter_stride();
        input_spec.padding_low =
            graph_node->as<CustomCall>()->get_padding_low();
        input_spec.padding_high =
            graph_node->as<CustomCall>()->get_padding_high();

        std::string output_tensor_name =
            EnvVar::is_true("TF_MONONN_ENABLED")
                ? graph_node->as<CustomCall>()->get_conv_output_GTE_node_name()
                :  // Output of conv is a tuple
                graph_node->get_name();

        Tensor output(output_tensor_name, graph_node->get_output_spec(0));

        std::shared_ptr<ConvImpl> conv_impl = std::make_shared<ConvImpl>(
            cuda_context, input_spec, *cutlass_config, *conv_backend_config,
            output);
        graph_node->set_implementation(conv_impl);
      } else {
        LOG(FATAL) << "Unsupported";
      }
    } else {
      if (graph_node->get_type() == OpType::global_sync ||
          graph_node->get_type() == OpType::get_tuple_element ||
          graph_node->get_type() == OpType::parameter ||
          graph_node->get_type() == OpType::constant) {
        auto impl = graph_node->generate_candidate_implementation(
            cuda_context, LocalityTier::kT0)[0];
        graph_node->set_implementation(impl);
      } else {
        LOG(FATAL) << "Un-Clustered node " << graph_node_name;
      }
    }
  }

  return true;
}
}  // namespace optimization
}  // namespace mononn_engine