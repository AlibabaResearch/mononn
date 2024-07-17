load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//mononn_engine/core/op:define_op.bzl", "define_all_cluster_op", "define_all_operators")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "op_type",
    srcs = ["op_type.cc"],
    hdrs = ["op_type.h"],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    hdrs = ["op.h"],
    deps = [
        "//mononn_engine/core/common:ilp_node_interface",
        "//mononn_engine/core/common:pointer_convert",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/gpu:functor",
        "//mononn_engine/core/gpu:multi_buffer",
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/core/op_annotation:cluster_type",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/op_impl:op_impl_base",
        "//mononn_engine/core/schedule",
        "//mononn_engine/core/schedule:loop",
        "//mononn_engine/core/tensor:math_op",
        "//mononn_engine/core/tensor:scalar",
        "//mononn_engine/core/tensor:tensor_spec",
    ],
)

define_all_operators([
    "abs",
    "add",
    "bitcast",
    "broadcast",
    "clamp",
    "compare",
    "concatenate",
    "constant",
    "convert",
    "convolution",
    "copy",
    "custom_call",
    "dynamic_slice",
    "dynamic_update_slice",
    "divide",
    "exp",
    "elewise_unary_op",
    "elewise_binary_op",
    "gather",
    "get_tuple_element",
    "global_sync",
    "iota",
    "log",
    "maximum",
    "minimum",
    "multiply",
    "negate",
    "sign",
    "output",
    "pad",
    "parameter",
    "reduce",
    "reduce_window",
    "reshape",
    "rsqrt",
    "select",
    "slice",
    "subtract",
    "tanh",
    "transpose",
    "transpose_smem",
    "tuple",
])

cc_library(
    name = "cluster_op",
    srcs = ["cluster_op.cc"],
    hdrs = ["cluster_op.h"],
    deps = [
        ":all_operators",
        "//mononn_engine/core/graph",
        "//mononn_engine/core/gpu:smem_manager",
        "//mononn_engine/core/op_impl:smem_prefetch_impl",
    ],
)

define_all_cluster_op([
    "cluster_elewise",
    "cluster_reduce",
    "cluster_gemm",
    "cluster_conv",
    "cluster_gemm_epilogue",
    "cluster_conv_epilogue",
])

# define_op("add")
# define_op("broadcast")
# define_op("reduce")

# cc_library(
#     name = "all_operators",
#     hdrs = [
#         "all_operators.h"
#     ],
#     srcs = [
#         "all_operators.cc"
#     ],
#     deps = [
#         ":op",
#         ":add",
#         ":broadcast",
#         ":reduce"
#     ]
# )
