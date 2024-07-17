load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
package(default_visibility = ["//mononn_engine:__subpackages__"])


cc_library(
    name = "tensor_shape",
    hdrs = ["tensor_shape.h"],
    srcs = ["tensor_shape.cc"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/helpers:string_helpers",
    ]
)

cc_library(
    name = "memory_layout",
    hdrs = ["memory_layout.h"],
    srcs = ["memory_layout.cc"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "dtype",
    hdrs = ["dtype.h"],
    srcs = ["dtype.cc"],
    deps = ["//mononn_engine/helpers:string_helpers"],
)

cc_library(
    name = "tensor_spec",
    hdrs = ["tensor_spec.h"],
    srcs = ["tensor_spec.cc"],
    deps = [
        ":dtype",
        ":tensor_shape",
        ":memory_layout",
        "//mononn_engine/helpers:macros",
    ],
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    srcs = ["tensor.cc"],
    deps = [
        ":tensor_spec"
    ],
)

cc_library(
    name = "scalar",
    hdrs = ["scalar.h"],
    srcs = ["scalar.cc"],
    deps = [
        ":dtype",
        "//mononn_engine/helpers:string_helpers"]
)

cc_library(
    name = "math_op",
    hdrs = ["math_op.h"],
    srcs = ["math_op.cc"],
    deps = [
        "//mononn_engine/helpers:macros"    
    ]
)