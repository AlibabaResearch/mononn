load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
)

cc_library(
    name = "graph_pass",
    srcs = ["graph_pass.cc"],
    hdrs = ["graph_pass.h"],
    deps = [
        ":common",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/op",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/op_annotation:op_attribute",
        "//mononn_engine/core/tensor",
    ],
)

#cc_library(
#    name = "clustered_graph_pass",
#    srcs = ["clustered_graph_pass.cc"],
#    hdrs = ["clustered_graph_pass.h"],
#    deps = [
#        "//mononn_engine/helpers:macros",
#        "//mononn_engine/core/graph:clustered_graph",
#        "//mononn_engine/core/op:cluster_op",
#        "//mononn_engine/core/op_annotation:cluster_type",
#        "//mononn_engine/core/tensor",
#    ],
#)

cc_library(
    name = "layout_simplification_pass",
    srcs = ["layout_simplification_pass.cc"],
    hdrs = ["layout_simplification_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "vectorization_pass",
    srcs = ["vectorization_pass.cc"],
    hdrs = ["vectorization_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "implementation_assignment_pass",
    srcs = ["implementation_assignment_pass.cc"],
    hdrs = ["implementation_assignment_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/gpu/cutlass:conv_backend_config",
        "//mononn_engine/core/gpu/cutlass:cutlass_config",
        "//mononn_engine/core/gpu/cutlass:gemm_backend_config",
        "//mononn_engine/core/op",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op:custom_call",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/op_impl:conv_impl",
        "//mononn_engine/core/op_impl:gemm_impl",
        "//mononn_engine/core/op_impl:op_impl_base",
        "//mononn_engine/core/schedule",
        "//mononn_engine/core/tensor",
        "//mononn_engine/proto:graph_specification_proto_cc",
    ],
)

cc_library(
    name = "initialize_smem_manager_pass",
    srcs = ["initialize_smem_manager_pass.cc"],
    hdrs = ["initialize_smem_manager_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "schedule_assignment_pass",
    srcs = ["schedule_assignment_pass.cc"],
    hdrs = ["schedule_assignment_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "smem_prefetch_pass",
    srcs = ["smem_prefetch_pass.cc"],
    hdrs = ["smem_prefetch_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op_annotation:auxiliary_impl_type",
        "//mononn_engine/core/op_impl:cache_prefetch_impl",
    ],
)

cc_library(
    name = "topology_simplification_pass",
    srcs = ["topology_simplification_pass.cc"],
    hdrs = ["topology_simplification_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op:concatenate",
        "//mononn_engine/core/op:slice",
    ],
)

cc_library(
    name = "trace_symbolic_index_pass",
    srcs = ["trace_symbolic_index_pass.cc"],
    hdrs = ["trace_symbolic_index_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "clustering_single_node_pass",
    srcs = ["clustering_single_node_pass.cc"],
    hdrs = ["clustering_single_node_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op:cluster_elewise",
        "//mononn_engine/core/op:cluster_reduce",
        "//mononn_engine/core/op_annotation:op_attribute",
    ],
)

cc_library(
    name = "element_wise_concatenation_pass",
    srcs = ["element_wise_concatenation_pass.cc"],
    hdrs = ["element_wise_concatenation_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/graph:cluster_util",
        "//mononn_engine/core/op:cluster_elewise",
        "//mononn_engine/core/op:cluster_reduce",
        "//mononn_engine/core/op_annotation:op_attribute",
    ],
)

cc_library(
    name = "memory_intensive_on_chip_fusion_pass",
    srcs = ["memory_intensive_on_chip_fusion_pass.cc"],
    hdrs = ["memory_intensive_on_chip_fusion_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "merge_independent_pass",
    srcs = ["merge_independent_pass.cc"],
    hdrs = ["merge_independent_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/graph:cluster_util",
    ],
)

cc_library(
    name = "merge_dependent_pass",
    srcs = ["merge_dependent_pass.cc"],
    hdrs = ["merge_dependent_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/graph:cluster_util",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "transpose_in_smem_pass",
    srcs = ["transpose_in_smem_pass.cc"],
    hdrs = ["transpose_in_smem_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "global_synchronization_assignment_pass",
    srcs = ["global_synchronization_assignment_pass.cc"],
    hdrs = ["global_synchronization_assignment_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "global_synchronization_elimination_pass",
    srcs = ["global_synchronization_elimination_pass.cc"],
    hdrs = ["global_synchronization_elimination_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op:global_sync",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "regional_synchronization_assignment_pass",
    srcs = ["regional_synchronization_assignment_pass.cc"],
    hdrs = ["regional_synchronization_assignment_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "locality_escalation_pass",
    srcs = ["locality_escalation_pass.cc"],
    hdrs = ["locality_escalation_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op_annotation:auxiliary_impl_type",
        "//mononn_engine/core/op_impl:parameter_read_reg_impl",
        "//mononn_engine/core/op_impl:parameter_shfl_impl",
        "//mononn_engine/core/op_impl:parameter_smem_impl",
    ],
)

cc_library(
    name = "explicit_output_pass",
    srcs = ["explicit_output_pass.cc"],
    hdrs = ["explicit_output_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op:output",
        "//mononn_engine/core/op_annotation:auxiliary_impl_type",
    ],
)

cc_library(
    name = "intra_op_reschedule_pass",
    srcs = ["intra_op_reschedule_pass.cc"],
    hdrs = ["intra_op_reschedule_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op:cluster_reduce",
    ],
)

cc_library(
    name = "access_pattern_analysis_pass",
    srcs = ["access_pattern_analysis_pass.cc"],
    hdrs = ["access_pattern_analysis_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "assign_cuda_context_pass",
    srcs = ["assign_cuda_context_pass.cc"],
    hdrs = ["assign_cuda_context_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "attribute_propagation_pass",
    srcs = ["attribute_propagation_pass.cc"],
    hdrs = ["attribute_propagation_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "buffer_assignment_pass",
    srcs = ["buffer_assignment_pass.cc"],
    hdrs = ["buffer_assignment_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/gpu:buffer_manager",
        "//mononn_engine/core/op",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op:constant",
        "//mononn_engine/core/op:op_type",
    ],
)

cc_library(
    name = "cache_prefetch_pass",
    srcs = ["cache_prefetch_pass.cc"],
    hdrs = ["cache_prefetch_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op_annotation:auxiliary_impl_type",
        "//mononn_engine/core/op_impl:cache_prefetch_impl",
    ],
)

cc_library(
    name = "cache_bypass_pass",
    srcs = ["cache_bypass_pass.cc"],
    hdrs = ["cache_bypass_pass.h"],
    deps = [
        ":graph_pass",
        "//mononn_engine/core/op_annotation:auxiliary_impl_type",
    ],
)

cc_library(
    name = "cache_temporal_access_pass",
    srcs = ["cache_temporal_access_pass.cc"],
    hdrs = ["cache_temporal_access_pass.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "pass_runner",
    srcs = ["pass_runner.cc"],
    hdrs = ["pass_runner.h"],
    deps = [
        ":graph_pass",
    ],
)

cc_library(
    name = "one_time_pass_runner",
    srcs = ["one_time_pass_runner.cc"],
    hdrs = ["one_time_pass_runner.h"],
    deps = [
        ":pass_runner",
    ],
)

cc_library(
    name = "run_greedy_pass_runner",
    srcs = ["run_greedy_pass_runner.cc"],
    hdrs = ["run_greedy_pass_runner.h"],
    deps = [
        ":pass_runner",
    ],
)

cc_library(
    name = "pass_manager",
    srcs = ["pass_manager.cc"],
    hdrs = ["pass_manager.h"],
    deps = [
        ":graph_pass",
        ":pass_runner",
    ],
)

cc_library(
    name = "optimization",
    srcs = ["optimization.cc"],
    hdrs = ["optimization.h"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/codegen:cuda_program",
        "//mononn_engine/codegen:graph_codegen",
        "//mononn_engine/core/edge",
        "//mononn_engine/core/gpu:defined",
        "//mononn_engine/core/gpu:functor",
        "//mononn_engine/core/gpu:memory",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/op:all_operators",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op_impl:gemm_impl",
        "//mononn_engine/core/schedule:vectorizer",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/core/tensor:tensor_spec",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/optimization:layout_simplification_pass",
        "//mononn_engine/parser:ir_parser",
        "//mononn_engine/parser:ir_parser_fused",
        "//mononn_engine/hlo",
        ":merge_independent_pass",
        ":merge_dependent_pass",
        ":pass_runner",
        ":pass_manager",
        ":one_time_pass_runner",
        ":run_greedy_pass_runner",
        ":global_synchronization_assignment_pass",
        ":global_synchronization_elimination_pass",
        ":locality_escalation_pass",
        ":regional_synchronization_assignment_pass",
        ":explicit_output_pass",
        ":intra_op_reschedule_pass",
        ":buffer_assignment_pass",
        ":clustering_single_node_pass",
        ":element_wise_concatenation_pass",
        ":cache_prefetch_pass",
        ":implementation_assignment_pass",
        ":initialize_smem_manager_pass",
        ":assign_cuda_context_pass",
    ],
)

cc_library(
    name = "optimization_runner",
    srcs = ["optimization_runner.cc"],
    hdrs = ["optimization_runner.h"],
    deps = [
        ":attribute_propagation_pass",
        ":buffer_assignment_pass",
        ":cache_prefetch_pass",
        ":clustering_single_node_pass",
        ":common",
        ":element_wise_concatenation_pass",
        ":explicit_output_pass",
        ":global_synchronization_assignment_pass",
        ":global_synchronization_elimination_pass",
        ":schedule_assignment_pass",
        ":implementation_assignment_pass",
        ":intra_op_reschedule_pass",
        ":locality_escalation_pass",
        ":merge_dependent_pass",
        ":merge_independent_pass",
        ":one_time_pass_runner",
        ":pass_manager",
        ":pass_runner",
        ":regional_synchronization_assignment_pass",
        ":run_greedy_pass_runner",
        ":vectorization_pass",
        ":access_pattern_analysis_pass",
        ":cache_temporal_access_pass",
        "//mononn_engine/proto:graph_specification_proto_cc",
        ":smem_prefetch_pass",
        ":initialize_smem_manager_pass",
        ":assign_cuda_context_pass",
        ":topology_simplification_pass",
        ":trace_symbolic_index_pass",
    ],
)