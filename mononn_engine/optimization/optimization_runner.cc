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

#include "mononn_engine/optimization/optimization_runner.h"

#include "mononn_engine/core/graph/cluster_util.h"
#include "mononn_engine/core/op/op.h"
#include "mononn_engine/helpers/macros.h"
#include "mononn_engine/optimization/access_pattern_analysis_pass.h"
#include "mononn_engine/optimization/assign_cuda_context_pass.h"
#include "mononn_engine/optimization/attribute_propagation_pass.h"
#include "mononn_engine/optimization/buffer_assignment_pass.h"
#include "mononn_engine/optimization/cache_prefetch_pass.h"
#include "mononn_engine/optimization/clustering_single_node_pass.h"
#include "mononn_engine/optimization/common.h"
#include "mononn_engine/optimization/element_wise_concatenation_pass.h"
#include "mononn_engine/optimization/explicit_output_pass.h"
#include "mononn_engine/optimization/global_synchronization_assignment_pass.h"
#include "mononn_engine/optimization/global_synchronization_elimination_pass.h"
#include "mononn_engine/optimization/implementation_assignment_pass.h"
#include "mononn_engine/optimization/initialize_smem_manager_pass.h"
#include "mononn_engine/optimization/intra_op_reschedule_pass.h"
#include "mononn_engine/optimization/locality_escalation_pass.h"
#include "mononn_engine/optimization/merge_dependent_pass.h"
#include "mononn_engine/optimization/merge_independent_pass.h"
#include "mononn_engine/optimization/one_time_pass_runner.h"
#include "mononn_engine/optimization/pass_manager.h"
#include "mononn_engine/optimization/pass_runner.h"
#include "mononn_engine/optimization/regional_synchronization_assignment_pass.h"
#include "mononn_engine/optimization/run_greedy_pass_runner.h"
#include "mononn_engine/optimization/schedule_assignment_pass.h"
#include "mononn_engine/optimization/smem_prefetch_pass.h"
#include "mononn_engine/optimization/topology_simplification_pass.h"
#include "mononn_engine/optimization/trace_symbolic_index_pass.h"
#include "mononn_engine/optimization/vectorization_pass.h"
// #include "mononn_engine/optimization/cache_bypass_pass.h"

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
using OpType = mononn_engine::core::op::OpType;
using Op = mononn_engine::core::op::Op;
using ClusterUtil = mononn_engine::core::graph::ClusterUtil;

void OptimizationRunner::run_group_pre_impl_assignment(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  LOG(INFO) << "==========Running optimization group pre implementation "
               "assignment==========";

  // static bool print_once = true;
  PassManager pass_manager(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           TopologySimplificationPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           ClusteringSingleNodePass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, RunGreedyPassRunner,
                           ElementWiseConcatenationPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           MergeIndependentPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, RunGreedyPassRunner,
                           MergeDependentPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           GlobalSynchronizationAssignmentPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           GlobalSynchronizationEliminationPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner, VectorizationPass);
  pass_manager.execute(graph);
}

void OptimizationRunner::run_group_impl_assignment(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
    const GraphSpecification* graph_specification) {
  LOG(INFO) << "==========Running optimization group implementation "
               "assignment==========";

  PassManager pass_manager(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           ScheduleAssignmentPass, graph_specification);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           ImplementationAssignmentPass, graph_specification);
  pass_manager.execute(graph);
}

void OptimizationRunner::run_group_impl_optimization(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context,
    const GraphSpecification* graph_specification) {
  LOG(INFO) << "==========Running optimization group implementation "
               "optimization==========";

  PassManager pass_manager(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner, ExplicitOutputPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           LocalityEscalationPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           AccessPatternAnalysisPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           AssignCUDAContextPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           InitializeSmemManagerPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           TraceSymbolicIndexPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner, SmemPrefetchPass);
  // ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner, CacheBypassPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           RegionalSynchronizationAssignmentPass);

  for (auto const& cluster_node_name :
       graph->get_node_list_by_type(OpType::cluster)) {
    std::shared_ptr<Op> cluster_node = graph->get_node(cluster_node_name);
    if (cluster_node->is_cluster_elewise()) {
      int ilp_factor = graph_specification->cluster_elewise_spec()
                           .at(cluster_node_name)
                           .ilp_factor();
      if (ilp_factor == 1) continue;
      ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                               IntraOpReschedulePass, cluster_node_name,
                               ilp_factor);
    } else if (cluster_node->is_cluster_reduce()) {
      int ilp_factor = graph_specification->cluster_reduce_spec()
                           .at(cluster_node_name)
                           .ilp_factor();
      if (ilp_factor == 1) continue;
      ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                               IntraOpReschedulePass, cluster_node_name,
                               ilp_factor);
    } else {
      LOG(FATAL) << "Unsupported cluster type"
                 << cluster_node->get_cluster_type().to_string()
                 << " for cluster " << cluster_node->get_name();
    }
  }

  pass_manager.execute(graph);
}

void OptimizationRunner::run_group_buffer_assignment(
    Graph* graph, std::shared_ptr<CUDAContext> cuda_context) {
  LOG(INFO)
      << "==========Running optimization group buffer assignment==========";

  PassManager pass_manager(cuda_context);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           BufferAssignmentPass);
  ADD_PASS_TO_PASS_MANAGER(pass_manager, OneTimePassRunner,
                           AttributePropagationPass);
  pass_manager.execute(graph);
}
}  // namespace optimization
}  // namespace mononn_engine