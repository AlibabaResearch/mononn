load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        "//mononn_engine/core/context:index_trace_stamp",
        "//mononn_engine/core/context:index_tracer",
        "//mononn_engine/core/edge",
        "//mononn_engine/core/edge:control_edge",
        "//mononn_engine/core/op_annotation:op_attribute",
        "//mononn_engine/helpers:macros",
    ],
)

cc_library(
    name = "clustered_graph",
    srcs = ["clustered_graph.cc"],
    hdrs = ["clustered_graph.h"],
    deps = [
        ":graph",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/core/edge:edge_cluster_op",
        "//mononn_engine/core/op:all_cluster_operators",
    ],
)

cc_library(
    name = "cluster_util",
    srcs = ["cluster_util.cc"],
    hdrs = ["cluster_util.h"],
    deps = [
        ":graph",
        "//mononn_engine/core/op:cluster_elewise",
        "//mononn_engine/core/op:cluster_op",
        "//mononn_engine/core/op:cluster_reduce",
    ],
)
