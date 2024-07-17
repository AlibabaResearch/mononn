load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "function_invocation",
    hdrs = ["function_invocation.h"],
    srcs = ["function_invocation.cc"],
    deps = [
        "//mononn_engine/helpers:string_helpers"
    ]
)

cc_library(
    name = "cuda_invocation",
    hdrs = ["cuda_invocation.h"],
    srcs = ["cuda_invocation.cc"],
    deps = [
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/core/gpu:dim3",
        ":function_invocation"
    ]
)

cc_library(
    name = "using",
    hdrs = ["using.h"],
    srcs = ["using.cc"],
    deps = [
        "//mononn_engine/helpers:string_helpers"
    ]
)