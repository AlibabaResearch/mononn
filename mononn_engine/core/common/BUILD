load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "compile_output_type",
    hdrs = ["compile_output_type.h"],
)

cc_library(
    name = "concurrent_queue",
    hdrs = ["concurrent_queue.h"],
    srcs = ["concurrent_queue.cc"]
)

cc_library(
    name = "pointer_convert",
    srcs = ["pointer_convert.cc"],
    hdrs = ["pointer_convert.h"],
)

# Deprecated
#cc_library(
#    name = "ilp_graph_interface",
#    srcs = ["ilp_graph_interface.cc"],
#    hdrs = ["ilp_graph_interface.h"],
#    deps = [
#        "//mononn_engine/core/context:index_trace_stamp",
#        "//mononn_engine/helpers",
#    ],
#)

cc_library(
    name = "ilp_node_interface",
    srcs = ["ilp_node_interface.cc"],
    hdrs = ["ilp_node_interface.h"],
    deps = [
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "ilp_node_impl_interface",
    srcs = ["ilp_node_impl_interface.cc"],
    hdrs = ["ilp_node_impl_interface.h"],
    deps = [
        ":ilp_node_interface",
        "//mononn_engine/core/context:index_trace_stamp",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "proto_converter",
    srcs = ["proto_converter.cc"],
    hdrs = ["proto_converter.h"],
    deps = [
    ],
)
