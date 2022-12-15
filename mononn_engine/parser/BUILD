load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "hlo_module_dumper",
    srcs = ["hlo_module_dumper.cc"],
    hdrs = ["hlo_module_dumper.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:ops",
        "//mononn_engine/cnpy",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "ir_parser",
    srcs = ["ir_parser.cc"],
    hdrs = ["ir_parser.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_runner",
        "//mononn_engine/core/graph",
    ],
)

tf_cc_binary(
    name = "tf2xla_parser_main",
    srcs = ["tf2xla_parser_main.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
        "@org_tensorflow//tensorflow/compiler/tf2xla",
        "@org_tensorflow//tensorflow/core",
        "@org_tensorflow//tensorflow/core:all_kernels",
        "@org_tensorflow//tensorflow/core:direct_session",
        "@org_tensorflow//tensorflow/core:ops",
        "@org_tensorflow//tensorflow/core/nccl:nccl_lib",
        "//mononn_engine/helpers",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cuda_platform_id",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_library(
    name = "ir_parser_fused",
    srcs = ["ir_parser_fused.cc"],
    hdrs = ["ir_parser_fused.h"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_parser",
        # "@org_tensorflow//tensorflow/compiler/xla/service:hlo_runner",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/graph:cluster_util",
        "//mononn_engine/core/graph:clustered_graph",
        "//mononn_engine/core/op_annotation:op_attribute",
        "//mononn_engine/codegen:reduction_functor_generator",
    ],
)
