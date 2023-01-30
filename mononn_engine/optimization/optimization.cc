#include "mononn_engine/optimization/optimization.h"

#include "mononn_engine/codegen/graph_codegen.h"
#include "mononn_engine/config/config.h"
#include "mononn_engine/core/gpu/dim3.h"
#include "mononn_engine/core/graph/graph.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/core/schedule/schedule.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/hlo/hlo.h"
#include "mononn_engine/optimization/buffer_assignment_pass.h"
#include "mononn_engine/optimization/clustering_single_node_pass.h"
#include "mononn_engine/optimization/element_wise_concatenation_pass.h"
#include "mononn_engine/optimization/explicit_output_pass.h"
#include "mononn_engine/optimization/global_synchronization_assignment_pass.h"
#include "mononn_engine/optimization/global_synchronization_elimination_pass.h"
#include "mononn_engine/optimization/intra_op_reschedule_pass.h"
#include "mononn_engine/optimization/locality_escalation_pass.h"
#include "mononn_engine/optimization/merge_dependent_pass.h"
#include "mononn_engine/optimization/merge_independent_pass.h"
#include "mononn_engine/optimization/one_time_pass_runner.h"
#include "mononn_engine/optimization/pass_manager.h"
#include "mononn_engine/optimization/pass_runner.h"
#include "mononn_engine/optimization/regional_synchronization_assignment_pass.h"
#include "mononn_engine/optimization/run_greedy_pass_runner.h"
#include "mononn_engine/parser/ir_parser_fused.h"

#define ADD_PASS_TO_PASS_MANAGER(pass_manager, pass_runner, pass_name, ...) \
  {                                                                         \
    std::unique_ptr<pass_name> __pass_##pass_name =                         \
        std::make_unique<pass_name>(__VA_ARGS__);                           \
    std::unique_ptr<PassRunner> __pass_runner_##pass_name =                 \
        std::make_unique<pass_runner>(std::move(__pass_##pass_name));       \
    pass_manager.add_runner(std::move(__pass_runner_##pass_name));          \
  }

namespace mononn_engine {
namespace optimization {
using CUDAProgram = mononn_engine::codegen::CUDAProgram;
using Config = mononn_engine::config::Config;
using Dim3 = mononn_engine::core::context::Dim3;
using Graph = mononn_engine::core::graph::Graph;
using CUDAContext = mononn_engine::core::context::CUDAContext;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using Op = mononn_engine::core::op::Op;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using Schedule = mononn_engine::core::schedule::Schedule;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using GraphCodegen = mononn_engine::codegen::GraphCodegen;
using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
using ElementWiseConcatenationPass =
    mononn_engine::optimization::ElementWiseConcatenationPass;

std::unique_ptr<CUDAProgram> Optimization::optimize() {
  std::shared_ptr<Graph> graph =
      mononn_engine::parser::IRParserFused::from_file(Config::get()->hlo_file);

  Dim3 grid_dim(108 * 2, 1, 1);
  Dim3 block_dim(128, 1, 1);
  std::string stream = "cuda_stream";
  std::shared_ptr<CUDAContext> cuda_context = std::make_shared<CUDAContext>(
      CUDAContext::get_cuda_context(grid_dim, block_dim, stream));
  LOG(INFO) << cuda_context->cuda_runtime_context.to_string();
  LOG(INFO) << cuda_context->cuda_device_context.to_string();

  PassManager pm_before_impl(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
                           ClusteringSingleNodePass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
                           ElementWiseConcatenationPass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
                           MergeIndependentPass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
                           MergeIndependentPass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
                           MergeDependentPass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
                           GlobalSynchronizationAssignmentPass);
  ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
                           GlobalSynchronizationEliminationPass);
  pm_before_impl.execute(graph.get());

  for (auto const cluster_node_name : graph->get_node_list()) {
    std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);

    if (cluster_node->get_type() == OpType::cluster) {
      for (auto const node_name :
           std::static_pointer_cast<ClusterOp>(cluster_node)
               ->get_graph()
               ->get_node_list()) {
        std::shared_ptr<Op> node =
            std::static_pointer_cast<ClusterOp>(cluster_node)
                ->get_graph()
                ->get_node(node_name);

        LocalityTier::Tier tier;
        if (node->get_type() == OpType::custom_call)
          tier = LocalityTier::kT3;
        else if (node->get_type() == OpType::reduce)
          tier = LocalityTier::kT1;
        else
          tier = LocalityTier::kT0;

        std::vector<std::shared_ptr<OpImplBase>> impl_list =
            node->generate_candidate_implementation(cuda_context, tier);

        node->set_implementation(impl_list[0]);
      }

      if (cluster_node->is_cluster_elewise()) {
        Schedule available_schedule =
            std::static_pointer_cast<ClusterOp>(cluster_node)
                ->construct_schedule(LocalityTier::kT0);
        std::static_pointer_cast<ClusterOp>(cluster_node)
            ->set_schedule(available_schedule);
      } else {
        Schedule available_schedule =
            std::static_pointer_cast<ClusterOp>(cluster_node)
                ->construct_schedule(LocalityTier::kT1);
        std::static_pointer_cast<ClusterOp>(cluster_node)
            ->set_schedule(available_schedule);
      }

    } else {
      LocalityTier::Tier tier;
      if (cluster_node->get_type() == OpType::custom_call)
        tier = LocalityTier::kT3;
      else if (cluster_node->get_type() == OpType::reduce)
        tier = LocalityTier::kT1;
      else
        tier = LocalityTier::kT0;

      std::vector<std::shared_ptr<OpImplBase>> impl_list =
          cluster_node->generate_candidate_implementation(cuda_context, tier);

      if (cluster_node->get_type() == OpType::custom_call) {
        bool selected = false;
        for (auto const& impl : impl_list) {
          if (impl->as<GemmImpl>()->get_cutlass_config().ThreadBlockShape.m() ==
                  64 &&
              impl->as<GemmImpl>()->get_cutlass_config().ThreadBlockShape.n() ==
                  64 &&
              impl->as<GemmImpl>()->get_cutlass_config().ThreadBlockShape.k() ==
                  32 &&
              impl->as<GemmImpl>()->get_cutlass_config().stages == 4) {
            LOG(WARNING) << "This should be removed in future";
            cluster_node->set_implementation(impl);
            selected = true;
            break;
          }
        }

        if (!selected) LOG(FATAL) << "";
      } else {
        cluster_node->set_implementation(impl_list[0]);
      }
    }
  }

  PassManager pm_after_impl(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           ExplicitOutputPass);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           LocalityEscalationPass);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           RegionalSynchronizationAssignmentPass);

  //        for (auto const &cluster_node_name :
  //        graph->get_node_list_by_type(OpType::cluster)) {
  //            std::shared_ptr<Op> cluster_node =
  //            graph->get_node(cluster_node_name);
  //
  //            if (cluster_node->is_cluster_reduce()) {
  //                ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //                IntraOpReschedulePass, cluster_node_name, 4);
  //            }
  //        }

  // bert tiny
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass, "fusion_5", 8);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass, "fusion_23", 8);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass, "fusion_11", 2);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass, "fusion_29", 2);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass,
                           "fusion_17_MergeIndependentPass_fusion_16", 2);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass,
                           "fusion_35_MergeIndependentPass_fusion_36", 2);

  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_45_MergeDependentPass_fusion_26_MergeDependentPass_fusion_24", 4);
  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_46_MergeDependentPass_fusion_20_MergeDependentPass_fusion_18", 4);
  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           IntraOpReschedulePass,
                           "fusion_47_MergeDependentPass_fusion_2", 4);
  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_48_MergeDependentPass_fusion_8_MergeDependentPass_fusion_6", 4);
  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_49_MergeDependentPass_fusion_14_MergeDependentPass_fusion_13", 4);
  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_50_MergeDependentPass_fusion_32_MergeDependentPass_fusion_31", 4);
  ADD_PASS_TO_PASS_MANAGER(
      pm_after_impl, OneTimePassRunner, IntraOpReschedulePass,
      "fusion_51_MergeDependentPass_fusion_39_MergeDependentPass_fusion_37", 4);

  // Bert base
  //        ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass, "fusion_204", 16);
  //        ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass, "fusion_210", 16);
  //        ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass, "fusion_215", 16);
  //        ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass,
  //        "fusion_271_MergeDependentPass_fusion_219_MergeDependentPass_fusion_217",
  //        8); ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass,
  //        "fusion_270_MergeDependentPass_fusion_213_MergeDependentPass_fusion_212",
  //        4); ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass,
  //        "fusion_236_MergeDependentPass_fusion_201_MergeDependentPass_fusion_199",
  //        8); ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //        IntraOpReschedulePass,
  //        "fusion_235_MergeDependentPass_fusion_207_MergeDependentPass_fusion_205",
  //        8);

  ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
                           BufferAssignmentPass);
  pm_after_impl.execute(graph.get());

  LOG(INFO) << "Graph summary after optimization";
  LOG(INFO) << graph->summary();
  GraphCodegen::Params params;
  params.cuda_context = cuda_context;
  params.graph = graph.get();
  params.kernel_namel = "mononn_kernel";
  params.argument_list = {Config::get()->onefuser_buffer_name};
  auto cuda_program = GraphCodegen::generate(params);

  return std::move(cuda_program);
}
}  // namespace optimization
}  // namespace mononn_engine
