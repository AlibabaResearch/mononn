load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "graph_tuner",
    srcs = ["graph_tuner.cc"],
    hdrs = ["graph_tuner.h"],
    deps = [
        "//mononn_engine/core/graph",
        "//mononn_engine/helpers",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "//mononn_engine/tuning/profiler:parallel_profiling_queue",
    ],
)

cc_library(
    name = "graph_cached_tuner",
    srcs = ["graph_cached_tuner.cc"],
    hdrs = ["graph_cached_tuner.h"],
    deps = [
        "//mononn_engine/core/graph",
        "//mononn_engine/helpers",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "//mononn_engine/tuning/profiler:parallel_profiling_queue",
    ],
)

cc_library(
    name = "tuning_space_generator",
    srcs = ["tuning_space_generator.cc"],
    hdrs = ["tuning_space_generator.h"],
    deps = [
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/graph",
        "//mononn_engine/helpers",
        "//mononn_engine/optimization:clustering_single_node_pass",
        "//mononn_engine/optimization:global_synchronization_assignment_pass",
        "//mononn_engine/optimization:global_synchronization_elimination_pass",
        "//mononn_engine/optimization:intra_op_reschedule_pass",
        "//mononn_engine/optimization:merge_dependent_pass",
        "//mononn_engine/optimization:merge_independent_pass",
        "//mononn_engine/optimization:one_time_pass_runner",
        "//mononn_engine/optimization:pass_manager",
        "//mononn_engine/optimization:pass_runner",
        "//mononn_engine/optimization:run_greedy_pass_runner",
        "//mononn_engine/parser:ir_parser_fused",
        "//mononn_engine/proto:graph_specification_proto_cc",
    ],
)

cc_library(
    name = "options",
    srcs = ["options.cc"],
    hdrs = ["options.h"],
    deps = [],
)

tf_cc_binary(
    name = "tuner_main",
    srcs = ["tuner_main.cc"],
    deps = [
        ":graph_cached_tuner",
        ":graph_tuner",
        ":options",
        ":tuning_space_generator",
        "//mononn_engine/optimization:optimization_runner",
        "//mononn_engine/codegen:graph_specification_codegen",
        "//mononn_engine/codegen:graph_specification_codegen_main",
        "//mononn_engine/helpers",
        "//mononn_engine/parser:hlo_module_dumper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

tf_cc_binary(
    name = "perf_dumper_main",
    srcs = ["perf_dumper_main.cc"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "//mononn_engine/tuning/profiler:parallel_profiling_queue"
    ],
)
