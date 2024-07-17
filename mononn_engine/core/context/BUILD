load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])


cc_library(
    name = "cuda_runtime_context",
    srcs = ["cuda_runtime_context.cc"],
    hdrs = ["cuda_runtime_context.h"],
    deps = [
        "//mononn_engine/core/gpu:dim3",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:cuda_runtime_context_proto_cc",
    ],
)

cc_library(
    name = "cuda_device_context",
    srcs = ["cuda_device_context.cc"],
    hdrs = ["cuda_device_context.h"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/gpu/cutlass:arch",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:cuda_device_context_proto_cc",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cuda_context",
    srcs = ["cuda_context.cc"],
    hdrs = ["cuda_context.h"],
    deps = [
        ":cuda_device_context",
        ":cuda_runtime_context",
        ":cuda_utils",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:cuda_context_proto_cc",
    ]
)

cc_library(
    name = "index_transform",
    srcs = ["index_transform.cc"],
    hdrs = ["index_transform.h"],
    deps = [
        "//mononn_engine/core/tensor:tensor_shape",
    ],
)

cc_library(
    name = "index_tracer",
    srcs = ["index_tracer.cc"],
    hdrs = ["index_tracer.h"],
    deps = [
        ":index_transform",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/op:broadcast",
        "//mononn_engine/core/op:concatenate",
        "//mononn_engine/core/op:constant",
        "//mononn_engine/core/op:copy",
        "//mononn_engine/core/op:dynamic_slice",
        "//mononn_engine/core/op:dynamic_update_slice",
        "//mononn_engine/core/op:gather",
        "//mononn_engine/core/op:pad",
        "//mononn_engine/core/op:reduce",
        "//mononn_engine/core/op:reduce_window",
        "//mononn_engine/core/op:reshape",
        "//mononn_engine/core/op:slice",
        "//mononn_engine/core/op:transpose",
    ],
)

cc_library(
    name = "index_trace_stamp",
    srcs = ["index_trace_stamp.cc"],
    hdrs = ["index_trace_stamp.h"],
    deps = [
        "//mononn_engine/helpers",
    ],
)

cuda_library(
    name = "cuda_utils",
    srcs = ["cuda_utils.cu.cc"],
    hdrs = ["cuda_utils.h"],
    deps = [
        "//mononn_engine/helpers",
    ]
)
