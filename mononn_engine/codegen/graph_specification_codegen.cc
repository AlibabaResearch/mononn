#include "mononn_engine/codegen/graph_specification_codegen.h"

#include "mononn_engine/config/config.h"
#include "mononn_engine/core/context/cuda_context.h"
#include "mononn_engine/core/gpu/cutlass/conv_backend_config.h"
#include "mononn_engine/core/gpu/cutlass/cutlass_config.h"
#include "mononn_engine/core/gpu/cutlass/gemm_backend_config.h"
#include "mononn_engine/core/op_impl/conv_impl.h"
#include "mononn_engine/core/op_impl/gemm_impl.h"
#include "mononn_engine/optimization/optimization_runner.h"
#include "mononn_engine/parser/ir_parser_fused.h"

namespace mononn_engine {
namespace codegen {
using Graph = mononn_engine::core::graph::Graph;
using CUDAContext = mononn_engine::core::context::CUDAContext;

using Op = mononn_engine::core::op::Op;
using OpType = mononn_engine::core::op::OpType;
using ClusterOp = mononn_engine::core::op::ClusterOp;
using CustomCall = mononn_engine::core::op::CustomCall;
using LocalityTier = mononn_engine::core::op_annotation::LocalityTier;
using OpImplBase = mononn_engine::core::op_impl::OpImplBase;
using Schedule = mononn_engine::core::schedule::Schedule;
using GemmImpl = mononn_engine::core::op_impl::GemmImpl;
using ConvImpl = mononn_engine::core::op_impl::ConvImpl;
using Config = mononn_engine::config::Config;
using CutlassConfig = mononn_engine::core::gpu::cutlass::CutlassConfig;
using GemmBackendConfig = mononn_engine::core::gpu::cutlass::GemmBackendConfig;
using ConvBackendConfig = mononn_engine::core::gpu::cutlass::ConvBackendConfig;
using Tensor = mononn_engine::core::tensor::Tensor;
using Dtype = mononn_engine::core::tensor::Dtype;
using OptimizationRunner = mononn_engine::optimization::OptimizationRunner;
// using BufferAssignmentPass =
// mononn_engine::optimization::BufferAssignmentPass; using PassManager =
// mononn_engine::optimization::PassManager; using PassRunner =
// mononn_engine::optimization::PassRunner; using AttributePropagationPass =
// mononn_engine::optimization::AttributePropagationPass; using
// RunGreedyPassRunner = mononn_engine::optimization::RunGreedyPassRunner; using
// OneTimePassRunner = mononn_engine::optimization::OneTimePassRunner; using
// MergeIndependentPass = mononn_engine::optimization::MergeIndependentPass;
// using MergeDependentPass = mononn_engine::optimization::MergeDependentPass;
// using GlobalSynchronizationAssignmentPass =
// mononn_engine::optimization::GlobalSynchronizationAssignmentPass; using
// GlobalSynchronizationEliminationPass =
// mononn_engine::optimization::GlobalSynchronizationEliminationPass; using
// ExplicitOutputPass = mononn_engine::optimization::ExplicitOutputPass; using
// LocalityEscalationPass = mononn_engine::optimization::LocalityEscalationPass;
// using RegionalSynchronizationAssignmentPass =
// mononn_engine::optimization::RegionalSynchronizationAssignmentPass; using
// IntraOpReschedulePass = mononn_engine::optimization::IntraOpReschedulePass;
// using ClusteringSingleNodePass =
// mononn_engine::optimization::ClusteringSingleNodePass; using
// ElementWiseConcatenationPass =
// mononn_engine::optimization::ElementWiseConcatenationPass; using
// ScheduleAssignmentPass = mononn_engine::optimization::ScheduleAssignmentPass;
// using ImplementationAssignmentPass =
// mononn_engine::optimization::ImplementationAssignmentPass; using
// VectorizationPass = mononn_engine::optimization::VectorizationPass; using
// CachePrefetchPass = mononn_engine::optimization::CachePrefetchPass; using
// AccessPatternAnalysisPass =
// mononn_engine::optimization::AccessPatternAnalysisPass; using PassName =
// mononn_engine::optimization::PassName; using SmemPrefetchPass =
// mononn_engine::optimization::SmemPrefetchPass; using AssignCUDAContextPass =
// mononn_engine::optimization::AssignCUDAContextPass; using
// InitializeSmemManagerPass =
// mononn_engine::optimization::InitializeSmemManagerPass; using
// TraceSymbolicIndexPass = mononn_engine::optimization::TraceSymbolicIndexPass;
// using CacheBypassPass = mononn_engine::optimization::CacheBypassPass;

std::unique_ptr<CUDAProgram> GraphSpecificationCodegen::generate(
    const tensorflow::mononn_extra::proto::GraphSpecification*
        graph_specification) {
  std::shared_ptr<Graph> graph =
      mononn_engine::parser::IRParserFused::from_hlo_module_proto_file(
          graph_specification->hlo_module_proto_file());
  std::shared_ptr<CUDAContext> cuda_context = std::make_shared<CUDAContext>();
  cuda_context->ParseFromProto(&graph_specification->cuda_context());

  OptimizationRunner::run_group_pre_impl_assignment(graph.get(), cuda_context);
  OptimizationRunner::run_group_impl_assignment(graph.get(), cuda_context,
                                                graph_specification);
  OptimizationRunner::run_group_impl_optimization(graph.get(), cuda_context,
                                                  graph_specification);
  OptimizationRunner::run_group_buffer_assignment(graph.get(), cuda_context);
  // PassManager pm_before_impl(cuda_context);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // ClusteringSingleNodePass); ADD_PASS_TO_PASS_MANAGER(pm_before_impl,
  // RunGreedyPassRunner, ElementWiseConcatenationPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, RunGreedyPassRunner,
  // MergeIndependentPass); ADD_PASS_TO_PASS_MANAGER(pm_before_impl,
  // RunGreedyPassRunner, MergeDependentPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // GlobalSynchronizationAssignmentPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_before_impl, OneTimePassRunner,
  // GlobalSynchronizationEliminationPass); pm_before_impl.execute(graph);

  // // for (auto const &cluster_node_name :
  // graph->get_node_list_by_type(OpType::cluster)) {
  // //     LOG(DEBUG) << "Cluster node: " << cluster_node_name;
  // //     auto cluster_node =
  // graph->get_node(cluster_node_name)->as<ClusterOp>();
  // //     std::vector<std::string> tags =
  // cluster_node->get_sub_cluster_tag_order();
  // //     std::vector<std::string> types =
  // cluster_node->get_sub_cluster_type_order();
  // //     for (int idx = 0; idx < (int)tags.size(); ++idx) {
  // //         LOG(DEBUG) << "\t" << tags[idx] << " " << types[idx];
  // //     }
  // // }

  // PassManager pm_impl_assignment(cuda_context);
  // ADD_PASS_TO_PASS_MANAGER(pm_impl_assignment, OneTimePassRunner,
  // VectorizationPass); ADD_PASS_TO_PASS_MANAGER(pm_impl_assignment,
  // OneTimePassRunner, ScheduleAssignmentPass, graph_specification);
  // ADD_PASS_TO_PASS_MANAGER(pm_impl_assignment, OneTimePassRunner,
  // ImplementationAssignmentPass, graph_specification);
  // pm_impl_assignment.execute(graph);

  // PassManager pm_after_impl(cuda_context);
  // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // ExplicitOutputPass); ADD_PASS_TO_PASS_MANAGER(pm_after_impl,
  // OneTimePassRunner, LocalityEscalationPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // AccessPatternAnalysisPass); ADD_PASS_TO_PASS_MANAGER(pm_after_impl,
  // OneTimePassRunner, AssignCUDAContextPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // InitializeSmemManagerPass); ADD_PASS_TO_PASS_MANAGER(pm_after_impl,
  // OneTimePassRunner, TraceSymbolicIndexPass);
  // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // SmemPrefetchPass);
  // // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // CacheBypassPass); ADD_PASS_TO_PASS_MANAGER(pm_after_impl,
  // OneTimePassRunner, RegionalSynchronizationAssignmentPass);

  // for (auto const &cluster_node_name :
  // graph->get_node_list_by_type(OpType::cluster)) {
  //     std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);
  //     if (cluster_node->is_cluster_elewise()) {
  //         int ilp_factor =
  //         graph_specification->cluster_elewise_spec().at(cluster_node_name).ilp_factor();
  //         if (ilp_factor == 1) continue;
  //         ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //         IntraOpReschedulePass, cluster_node_name, ilp_factor);
  //     } else if (cluster_node->is_cluster_reduce()) {
  //         int ilp_factor =
  //         graph_specification->cluster_reduce_spec().at(cluster_node_name).ilp_factor();
  //         if (ilp_factor == 1) continue;
  //         ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  //         IntraOpReschedulePass, cluster_node_name, ilp_factor);
  //     } else {
  //         LOG(FATAL) << "Unsupported cluster type" <<
  //         cluster_node->get_cluster_type().to_string() << " for cluster " <<
  //         cluster_node->get_name();
  //     }
  // }

  // ADD_PASS_TO_PASS_MANAGER(pm_after_impl, OneTimePassRunner,
  // BufferAssignmentPass); ADD_PASS_TO_PASS_MANAGER(pm_after_impl,
  // OneTimePassRunner, AttributePropagationPass); pm_after_impl.execute(graph);

  std::unordered_set<std::string> codegen_reject_list;
  codegen_reject_list.insert(graph_specification->codegen_reject_list().begin(),
                             graph_specification->codegen_reject_list().end());

  LOG(INFO) << "Graph summary after optimization";
  LOG(INFO) << graph->summary();
  GraphCodegen::Params params{cuda_context,
                              graph.get(),
                              codegen_reject_list,
                              "mononn_kernel",
                              {Config::get()->onefuser_buffer_name}};

  auto cuda_program = GraphCodegen::generate(params);

  for (auto& data_file : graph_specification->input_data_files()) {
    cuda_program->add_data_file(data_file);
  }

  return std::move(cuda_program);
}
}  // namespace codegen
}  // namespace mononn_engine