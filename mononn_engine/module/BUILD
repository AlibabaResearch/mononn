load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "mononn_module",
    hdrs = ["mononn_module.h"],
    srcs = ["mononn_module.cc"],
    deps = [
        "//mononn_engine/codegen:cuda_program",
        "//mononn_engine/parser:ir_parser_fused",
        "//mononn_engine/optimization:optimization_runner",
        "//mononn_engine/helpers:helpers",
        "//mononn_engine/codegen:graph_codegen",
        "//mononn_engine/tuning/profiler:subprocess",
        "//mononn_engine/proto:graph_specification_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
    ],
)

cc_library(
    name = "mononn_module_tuner",
    hdrs = ["mononn_module_tuner.h"],
    srcs = ["mononn_module_tuner.cc"],
    deps = [
        ":mononn_module",
        "//mononn_engine/tuning:tuning_space_generator",
        "//mononn_engine/optimization:optimization_runner",
        "//mononn_engine/parser:ir_parser_fused",
        "//mononn_engine/helpers:helpers",
        "//mononn_engine/codegen:compilation_threadpool",
        "//mononn_engine/core/common:concurrent_queue",
        "//mononn_engine/tuning/profiler:cupti_profiling_session",
        "//mononn_engine/tuning/profiler:timer",
        "@org_tensorflow//tensorflow/stream_executor/gpu:gpu_executor_header",
        "@org_tensorflow//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "@org_tensorflow//tensorflow/stream_executor:stream_executor",
        "@org_tensorflow//tensorflow/stream_executor/cuda:cuda_driver",
    ],
)