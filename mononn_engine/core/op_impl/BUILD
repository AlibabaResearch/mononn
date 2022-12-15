load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "reducer",
    srcs = ["reducer.cc"],
    hdrs = ["reducer.h"],
)

cc_library(
    name = "op_impl_base",
    srcs = ["op_impl_base.cc"],
    hdrs = ["op_impl_base.h"],
    deps = [
        "//mononn_engine/config",
        "//mononn_engine/core/common:ilp_node_impl_interface",
        "//mononn_engine/core/common:pointer_convert",
        "//mononn_engine/core/context:cuda_context",
        "//mononn_engine/core/context:index_trace_stamp",
        "//mononn_engine/core/gpu:buffer_manager",
        "//mononn_engine/core/gpu:functor",
        "//mononn_engine/core/gpu:memory",
        "//mononn_engine/core/op_annotation:locality_tier",
        "//mononn_engine/core/schedule",
        "//mononn_engine/core/schedule:loop",
        "//mononn_engine/core/semantic:function_invocation",
        "//mononn_engine/core/tensor",
        "//mononn_engine/core/tensor:dtype",
        "//mononn_engine/core/tensor:math_op",
        "//mononn_engine/core/tensor:scalar",
        "//mononn_engine/core/tensor:tensor_spec",
        "//mononn_engine/helpers:macros",
    ],
)

cc_library(
    name = "reduce_impl",
    srcs = ["reduce_impl.cc"],
    hdrs = ["reduce_impl.h"],
    deps = [
        ":op_impl_base",
        ":reducer",
        "//mononn_engine/core/gpu:limits",
        "//mononn_engine/codegen:reduction_functor_generator",
    ],
)

cc_library(
    name = "elewise_unary_impl",
    srcs = ["elewise_unary_impl.cc"],
    hdrs = ["elewise_unary_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "elewise_binary_impl",
    srcs = ["elewise_binary_impl.cc"],
    hdrs = ["elewise_binary_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "constant_impl",
    srcs = ["constant_impl.cc"],
    hdrs = ["constant_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/gpu:limits",
    ],
)

cc_library(
    name = "parameter_impl_base",
    srcs = ["parameter_impl_base.cc"],
    hdrs = ["parameter_impl_base.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "parameter_impl",
    srcs = ["parameter_impl.cc"],
    hdrs = ["parameter_impl.h"],
    deps = [
        ":parameter_impl_base",
        "//mononn_engine/core/op_annotation:op_attribute",
    ],
)

cc_library(
    name = "parameter_read_reg_impl",
    srcs = ["parameter_read_reg_impl.cc"],
    hdrs = ["parameter_read_reg_impl.h"],
    deps = [
        ":parameter_impl_base",
    ],
)

cc_library(
    name = "parameter_shfl_impl",
    srcs = ["parameter_shfl_impl.cc"],
    hdrs = ["parameter_shfl_impl.h"],
    deps = [
        ":parameter_impl_base",
    ],
)

cc_library(
    name = "parameter_smem_impl",
    srcs = ["parameter_smem_impl.cc"],
    hdrs = ["parameter_smem_impl.h"],
    deps = [
        ":parameter_impl_base",
    ],
)

cc_library(
    name = "output_impl_base",
    srcs = ["output_impl_base.cc"],
    hdrs = ["output_impl_base.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/helpers",
    ],
)

cc_library(
    name = "output_impl",
    srcs = ["output_impl.cc"],
    hdrs = ["output_impl.h"],
    deps = [
        ":output_impl_base",
    ],
)

cc_library(
    name = "output_reg_impl",
    srcs = ["output_reg_impl.cc"],
    hdrs = ["output_reg_impl.h"],
    deps = [
        ":output_impl_base",
    ],
)

cc_library(
    name = "broadcast_impl",
    srcs = ["broadcast_impl.cc"],
    hdrs = ["broadcast_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "cache_prefetch_impl",
    srcs = ["cache_prefetch_impl.cc"],
    hdrs = ["cache_prefetch_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/op_annotation:op_attribute",
    ],
)

cc_library(
    name = "gather_impl",
    srcs = ["gather_impl.cc"],
    hdrs = ["gather_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "gemm_impl",
    srcs = ["gemm_impl.cc"],
    hdrs = ["gemm_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/gpu/cutlass",
        "//mononn_engine/core/gpu/cutlass:cutlass_config",
        "//mononn_engine/core/gpu/cutlass:gemm_backend_config",
        "//mononn_engine/core/semantic:using",
        "//mononn_engine/helpers:json",
        "//mononn_engine/proto:cutlass_config_proto_cc",
        "//mononn_engine/proto:gemm_backend_config_proto_cc",
    ],
)

cc_library(
    name = "conv_impl",
    srcs = ["conv_impl.cc"],
    hdrs = ["conv_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/gpu/cutlass",
        "//mononn_engine/core/gpu/cutlass:conv2d_problem_size",
        "//mononn_engine/core/gpu/cutlass:conv_argument",
        "//mononn_engine/core/gpu/cutlass:conv_backend_config",
        "//mononn_engine/core/gpu/cutlass:cutlass_config",
        "//mononn_engine/core/semantic:using",
        "//mononn_engine/helpers:json",
        "//mononn_engine/proto:conv_backend_config_proto_cc",
        "//mononn_engine/proto:cutlass_config_proto_cc",
    ],
)

cc_library(
    name = "dynamic_slice_impl",
    srcs = ["dynamic_slice_impl.cc"],
    hdrs = ["dynamic_slice_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "dynamic_update_slice_impl",
    srcs = ["dynamic_update_slice_impl.cc"],
    hdrs = ["dynamic_update_slice_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "concatenate_impl",
    srcs = ["concatenate_impl.cc"],
    hdrs = ["concatenate_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/context:index_transform",
    ],
)

cc_library(
    name = "iota_impl",
    srcs = ["iota_impl.cc"],
    hdrs = ["iota_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "pad_impl",
    srcs = ["pad_impl.cc"],
    hdrs = ["pad_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "reshape_impl",
    srcs = ["reshape_impl.cc"],
    hdrs = ["reshape_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "slice_impl",
    srcs = ["slice_impl.cc"],
    hdrs = ["slice_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "smem_prefetch_impl",
    srcs = ["smem_prefetch_impl.cc"],
    hdrs = ["smem_prefetch_impl.h"],
    deps = [
        ":parameter_impl_base",
        "//mononn_engine/core/op_annotation:op_attribute",
    ],
)

cc_library(
    name = "transpose_impl",
    srcs = ["transpose_impl.cc"],
    hdrs = ["transpose_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "get_tuple_element_impl",
    srcs = ["get_tuple_element_impl.cc"],
    hdrs = ["get_tuple_element_impl.h"],
    deps = [
        ":op_impl_base",
        "//mononn_engine/core/gpu:multi_buffer",
    ],
)

cc_library(
    name = "clamp_impl",
    srcs = ["clamp_impl.cc"],
    hdrs = ["clamp_impl.h"],
    deps = [
        ":op_impl_base",
    ],
)

cc_library(
    name = "transpose_smem_impl",
    srcs = ["transpose_smem_impl.cc"],
    hdrs = ["transpose_smem_impl.h"],
    deps = [":op_impl_base"],
)

cc_library(
    name = "global_sync_impl",
    srcs = ["global_sync_impl.cc"],
    hdrs = ["global_sync_impl.h"],
    deps = [":op_impl_base"],
)
