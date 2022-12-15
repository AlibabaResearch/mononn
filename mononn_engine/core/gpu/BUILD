load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "defined",
    srcs = [
        "defined.cc",
        "defined.h",
    ],
    hdrs = ["defined.h"],
    deps = [
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "headers",
    srcs = ["headers.cc"],
    hdrs = ["headers.h"],
)

cc_library(
    name = "functor",
    srcs = ["functor.cc"],
    hdrs = ["functor.h"],
    deps = [
        ":defined",
        "//mononn_engine/config",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/core/tensor:math_op",
    ],
)

cc_library(
    name = "synchronization",
    srcs = ["synchronization.cc"],
    hdrs = ["synchronization.h"],
    deps = [
        ":defined",
    ],
)

cc_library(
    name = "reduction",
    srcs = ["reduction.cc"],
    hdrs = ["reduction.h"],
    deps = [
        ":defined",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/gpu:functor",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/semantic:function_invocation",
        "//mononn_engine/core/tensor",
        "//mononn_engine/core/tensor:tensor_spec",
    ],
)

cc_library(
    name = "dim3",
    srcs = ["dim3.cc"],
    hdrs = ["dim3.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:dim3_proto_cc",
    ],
)

cc_library(
    name = "memory",
    srcs = ["memory.cc"],
    hdrs = ["memory.h"],
    deps = [
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "buffer_manager",
    srcs = ["buffer_manager.cc"],
    hdrs = ["buffer_manager.h"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "limits",
    srcs = ["limits.cc"],
    hdrs = ["limits.h"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/tensor:dtype",
    ],
)

cc_library(
    name = "smem_manager",
    srcs = ["smem_manager.cc"],
    hdrs = ["smem_manager.h"],
    deps = [
        ":multi_buffer",
        "//mononn_engine/core/tensor:tensor_spec",
    ],
)

cc_library(
    name = "multi_buffer",
    srcs = ["multi_buffer.cc"],
    hdrs = ["multi_buffer.h"],
    deps = [
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/core/tensor:tensor_spec",
    ],
)
