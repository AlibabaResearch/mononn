"""Provides functionality for define operator."""

load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

elewise_unary_op_list = [
    "abs",
    "ceil",
    "exp",
    "tanh",
    "rsqrt",
    "convert",
    "clamp",
    "log",
    "sign",
    "negate",
]

elewise_binary_op_list = [
    "add",
    "compare",
    "divide",
    "maximum",
    "minimum",
    "multiply",
    "subtract",
    "select",
]

def define_op(name):
    deps = [":op"]
    if name in elewise_unary_op_list:
        deps.append(":elewise_unary_op")
    if name in elewise_binary_op_list:
        deps.append(":elewise_binary_op")

    # Will be removed in future
    if name == "bitcast" or name == "copy":
        deps.append("//mononn_engine/core/op_impl:elewise_unary_impl")

    if name == "parameter":
        deps.append("//mononn_engine/core/op_impl:parameter_impl_base")
        deps.append("//mononn_engine/core/op_impl:parameter_impl")

    if name == "broadcast":
        deps.append("//mononn_engine/core/op_impl:broadcast_impl")

    if name == "constant":
        deps.append("//mononn_engine/core/op_impl:constant_impl")

    if name == "gather":
        deps.append("//mononn_engine/core/op_impl:gather_impl")

    if name == "concatenate":
        deps.append("//mononn_engine/core/op_impl:concatenate_impl")

    if name == "get_tuple_element":
        deps.append("//mononn_engine/core/op_impl:get_tuple_element_impl")

    if name == "iota":
        deps.append("//mononn_engine/core/op_impl:iota_impl")

    if name == "pad":
        deps.append("//mononn_engine/core/op_impl:pad_impl")
        deps.append("//mononn_engine/core/context:index_transform")

    if name == "reshape":
        deps.append("//mononn_engine/core/op_impl:reshape_impl")

    if name == "slice":
        deps.append("//mononn_engine/core/op_impl:slice_impl")

    if name == "transpose":
        deps.append("//mononn_engine/core/op_impl:transpose_impl")
    if name == "transpose_smem":
        deps.append("//mononn_engine/core/op_impl:transpose_smem_impl")

    if name == "elewise_unary_op":
        deps.append("//mononn_engine/core/op_impl:elewise_unary_impl")
    if name == "elewise_binary_op":
        deps.append("//mononn_engine/core/op_impl:elewise_binary_impl")

    if name == "reduce":
        deps.append("//mononn_engine/core/op_impl:reducer")
        deps.append("//mononn_engine/core/op_impl:reduce_impl")
        deps.append("//mononn_engine/codegen:reduction_functor_generator")
    if name == "reduce_window":
        deps.append("//mononn_engine/codegen:reduction_functor_generator")
        
    if name == "custom_call":
        deps.append("//mononn_engine/core/op_impl:gemm_impl")
        deps.append("//mononn_engine/core/op_impl:conv_impl")
    if name == "clamp":
        deps.append("//mononn_engine/core/op_impl:clamp_impl")
    if name == "global_sync":
        deps.append("//mononn_engine/core/op_impl:global_sync_impl")
    if name == "output":
        deps.append("//mononn_engine/core/op_impl:output_impl")
        deps.append("//mononn_engine/core/op_impl:output_reg_impl")

    if name == "dynamic_slice":
        deps.append("//mononn_engine/core/op_impl:dynamic_slice_impl")
    if name == "dynamic_update_slice":
        deps.append("//mononn_engine/core/op_impl:dynamic_update_slice_impl")

    cc_library(
        name = name,
        hdrs = [
            name + ".h",
        ],
        srcs = [
            name + ".cc",
        ],
        deps = deps,
    )

def define_all_operators(op_list):
    for op in op_list:
        define_op(op)

    cc_library(
        name = "all_operators",
        hdrs = [
            "all_operators.h",
        ],
        srcs = [
            "all_operators.cc",
        ],
        deps = [
            ":" + op
            for op in op_list
        ],
    )

def define_cluster_op(name):
    deps = [
        ":cluster_op",
        "//mononn_engine/core/schedule:schedule_factory",
        "//mononn_engine/core/context:index_tracer",
        "//mononn_engine/core/op_annotation:op_attribute",
        "//mononn_engine/codegen:cuda_emitter",
    ]

    cc_library(
        name = name,
        hdrs = [
            name + ".h",
        ],
        srcs = [
            name + ".cc",
        ],
        deps = deps,
    )

def define_all_cluster_op(op_list):
    for cluster_op in op_list:
        define_cluster_op(cluster_op)

    cc_library(
        name = "all_cluster_operators",
        hdrs = [
            "all_cluster_operators.h",
        ],
        srcs = [
            "all_cluster_operators.cc",
        ],
        deps = [
            ":" + op
            for op in op_list
        ],
    )
