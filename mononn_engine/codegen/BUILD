load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary",)
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "model_data",
    srcs = ["model_data.cc"],
    hdrs = ["model_data.h"],
    deps = [
        "//mononn_engine/cnpy",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/helpers",
        #        "//third_party/eigen3/Eigen/Core",
    ],
)

cc_library(
    name = "cuda_program",
    srcs = [
        "cuda_program.cc",
    ],
    hdrs = [
        "cuda_program.h",
    ],
    linkopts = [
        "-lstdc++fs",
    ],
    deps = [
        ":model_data",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/config",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/core/common:compile_output_type",
        "//mononn_engine/helpers:path",
    ],
)

cc_library(
    name = "cluster_codegen",
    srcs = [
        "cluster_codegen.cc",
    ],
    hdrs = [
        "cluster_codegen.h",
    ],
    deps = [
        ":cuda_program",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/context:index_tracer",
        "//mononn_engine/core/op:cluster_op",
    ],
)

cc_library(
    name = "compilation_threadpool",
    srcs = ["compilation_threadpool.cc"],
    hdrs = ["compilation_threadpool.h"],
    deps = [
        "//mononn_engine/tuning/profiler:thread_pool",
        "//mononn_engine/core/common:concurrent_queue",
        "//mononn_engine/core/graph:graph",
        "//mononn_engine/frontend:mononn_module",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service/gpu:buffer_allocations",
    ]
)

cc_library(
    name = "cuda_emitter",
    srcs = ["cuda_emitter.cc"],
    hdrs = ["cuda_emitter.h"],
    deps = [
        "//mononn_engine/core/graph:graph",
        "//mononn_engine/core/op:broadcast",
        "//mononn_engine/core/op:compare",
        "//mononn_engine/core/op:constant",
        "//mononn_engine/core/op:convert",
        "//mononn_engine/core/op:multiply",
        "//mononn_engine/core/op:gather",
        "//mononn_engine/core/op:parameter",
        "//mononn_engine/core/op:reduce_window",
        "//mononn_engine/core/context:index_trace_stamp",
        "//mononn_engine/core/gpu:functor",
        "//mononn_engine/core/semantic:function_invocation",
        "//mononn_engine/core/gpu:memory",
        "//mononn_engine/core/gpu:buffer_manager",
        "//mononn_engine/helpers",
    ]
)

cc_library(
    name = "node_codegen",
    srcs = [
        "node_codegen.cc",
    ],
    hdrs = [
        "node_codegen.h",
    ],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/op",
        "//mononn_engine/core/op:constant",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/op_impl:op_impl_base",
    ],
)

cc_library(
    name = "graph_codegen",
    srcs = [
        "graph_codegen.cc",
    ],
    hdrs = [
        "graph_codegen.h",
    ],
    deps = [
        ":cluster_codegen",
        ":cuda_program",
        ":host_codegen",
        ":node_codegen",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/gpu:headers",
        "//mononn_engine/core/gpu:limits",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/op",
        "@org_tensorflow//tensorflow/compiler/xla/service:buffer_assignment",
        "@org_tensorflow//tensorflow/compiler/xla/service/gpu:buffer_allocations",
    ],
)

cc_library(
    name = "graph_specification_codegen",
    srcs = ["graph_specification_codegen.cc"],
    hdrs = ["graph_specification_codegen.h"],
    deps = [
        ":graph_codegen",
        "//mononn_engine/core/op_impl:conv_impl",
        "//mononn_engine/core/op_impl:gemm_impl",
        "//mononn_engine/parser:ir_parser_fused",
        "//mononn_engine/optimization:optimization_runner",
    ],
)

cc_library(
    name = "host_codegen",
    srcs = [
        "host_codegen.cc",
    ],
    hdrs = [
        "host_codegen.h",
    ],
    deps = [
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/op",
        "//mononn_engine/core/semantic:cuda_invocation",
        "//mononn_engine/helpers:path",
    ],
)

tf_cc_binary(
    name = "graph_specification_codegen_main",
    srcs = ["graph_specification_codegen_main.cc"],
    deps = [
        ":graph_specification_codegen",
        "//mononn_engine/helpers",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)


cc_library(
    name = "reduction_functor_generator",
    srcs = ["reduction_functor_generator.cc"],
    hdrs = ["reduction_functor_generator.h"],
    deps = [
        "//mononn_engine/helpers",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
    ],
)
