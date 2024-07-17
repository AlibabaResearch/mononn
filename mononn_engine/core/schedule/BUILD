load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "loop",
    srcs = ["loop.cc"],
    hdrs = ["loop.h"],
    deps = [
        "//mononn_engine/core/tensor:math_op",
        "//mononn_engine/core/tensor:scalar",
        "//mononn_engine/core/tensor:tensor_shape",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "schedule",
    srcs = ["schedule.cc"],
    hdrs = ["schedule.h"],
    deps = [
        ":loop",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/tensor:tensor_shape",
    ],
)

cc_library(
    name = "schedule_factory",
    srcs = ["schedule_factory.cc"],
    hdrs = ["schedule_factory.h"],
    deps = [
        ":loop",
        ":schedule",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/gpu:defined",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/core/tensor:scalar",
        "//mononn_engine/core/tensor:tensor_shape",
    ],
)

cc_library(
    name = "vectorizer",
    srcs = ["vectorizer.cc"],
    hdrs = ["vectorizer.h"],
    deps = [
        "schedule",
        "//mononn_engine/core/op:cluster_op",
    ],
)
