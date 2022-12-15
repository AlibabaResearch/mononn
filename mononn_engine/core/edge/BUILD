load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "edge",
    srcs = ["edge.cc"],
    hdrs = ["edge.h"],
    deps = [
        "//mononn_engine/core/gpu:synchronization",
        "//mononn_engine/core/op:all_operators",
    ],
)

cc_library(
    name = "edge_cluster_op",
    srcs = ["edge_cluster_op.cc"],
    hdrs = ["edge.h"],
    deps = [
        ":edge",
        "//mononn_engine/core/op:cluster_op",
    ],
)

cc_library(
    name = "control_edge",
    srcs = ["control_edge.cc"],
    hdrs = ["control_edge.h"],
    deps = [
        "//mononn_engine/core/op",
    ],
)
