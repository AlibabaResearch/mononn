load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "arch",
    srcs = ["arch.cc"],
    hdrs = ["arch.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/helpers:macros",
        "//mononn_engine/proto:arch_proto_cc",
    ],
)

cc_library(
    name = "layout",
    srcs = ["layout.cc"],
    hdrs = ["layout.h"],
)

cc_library(
    name = "gemm_shape",
    srcs = ["gemm_shape.cc"],
    hdrs = ["gemm_shape.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:gemm_shape_proto_cc",
    ],
)

cc_library(
    name = "cutlass_config",
    srcs = ["cutlass_config.cc"],
    hdrs = ["cutlass_config.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/core/gpu/cutlass:arch",
        "//mononn_engine/core/gpu/cutlass:gemm_shape",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:cutlass_config_proto_cc",
        "//mononn_engine/proto:gemm_shape_proto_cc",
    ],
)

cc_library(
    name = "gemm_backend_config",
    srcs = ["gemm_backend_config.cc"],
    hdrs = ["gemm_backend_config.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:gemm_backend_config_proto_cc",
    ],
)

cc_library(
    name = "conv_backend_config",
    srcs = ["conv_backend_config.cc"],
    hdrs = ["conv_backend_config.h"],
    deps = [
        "//mononn_engine/core/common:proto_converter",
        "//mononn_engine/helpers:string_helpers",
        "//mononn_engine/proto:conv_backend_config_proto_cc",
    ],
)

cc_library(
    name = "gemm_coord",
    srcs = [
        "gemm_coord.cc",
    ],
    hdrs = [
        "gemm_coord.h",
    ],
    deps = [
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "gemm_universal_mode",
    srcs = [
        "gemm_universal_mode.cc",
    ],
    hdrs = [
        "gemm_universal_mode.h",
    ],
)

cc_library(
    name = "iterator_algorithm",
    srcs = [
        "iterator_algorithm.cc",
    ],
    hdrs = [
        "iterator_algorithm.h",
    ],
)

cc_library(
    name = "stride_support",
    srcs = [
        "stride_support.cc",
    ],
    hdrs = [
        "stride_support.h",
    ],
)

cc_library(
    name = "gemm_argument",
    srcs = [
        "gemm_argument.cc",
    ],
    hdrs = [
        "gemm_argument.h",
    ],
    deps = [
        ":gemm_coord",
        ":gemm_universal_mode",
    ],
)

cc_library(
    name = "conv2d_problem_size",
    srcs = [
        "conv2d_problem_size.cc",
    ],
    hdrs = [
        "conv2d_problem_size.h",
    ],
    deps = [
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "conv_argument",
    srcs = [
        "conv_argument.cc",
    ],
    hdrs = [
        "conv_argument.h",
    ],
    deps = [
        ":conv2d_problem_size",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "swizzle",
    srcs = ["swizzle.cc"],
    hdrs = ["swizzle.h"],
)

cc_library(
    name = "shared_storage",
    srcs = ["shared_storage.cc"],
    hdrs = ["shared_storage.h"],
    deps = [":gemm_shape"],
)

cc_library(
    name = "tile_description",
    srcs = ["tile_description.cc"],
    hdrs = ["tile_description.h"],
    deps = [
        ":arch",
        ":gemm_shape",
        "//mononn_engine/config",
        "//mononn_engine/core/tensor:dtype",
    ],
)

cc_library(
    name = "cutlass",
    hdrs = ["cutlass.h"],
    deps = [
        ":arch",
        ":conv2d_problem_size",
        ":conv_argument",
        ":conv_backend_config",
        ":gemm_argument",
        ":gemm_coord",
        ":gemm_shape",
        ":gemm_universal_mode",
        ":iterator_algorithm",
        ":layout",
        ":shared_storage",
        ":stride_support",
        ":swizzle",
        ":tile_description",
    ],
)
