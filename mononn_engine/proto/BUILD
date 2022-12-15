load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")


package(default_visibility = ["//mononn_engine:__subpackages__"])

# proto_library(
#     name = "arch_proto",
#     srcs = ["arch.proto"],
#     deps = [
#         "@com_google_protobuf//:any_proto",
#     ]
# )

# proto_library(
#     name = "gemm_shape_proto",
#     srcs = ["gemm_shape.proto"],
# )

# proto_library(
#     name = "cutlass_config_proto",
#     srcs = ["cutlass_config.proto"],
#     deps = [
#         ":gemm_shape_proto",
#         ":arch_proto"
#     ]
# )

# proto_library(
#     name = "gemm_backend_config_proto",
#     srcs = ["gemm_backend_config.proto"],
# )

# proto_library(
#     name = "conv_backend_config_proto",
#     srcs = ["conv_backend_config.proto"],
# )

# proto_library(
#     name = "cluster_elewise_specification_proto",
#     srcs = ["cluster_elewise_specification.proto"],
# )

# proto_library(
#     name = "cluster_reduce_specification_proto",
#     srcs = ["cluster_reduce_specification.proto"],
# )

# proto_library(
#     name = "gemm_specification_proto",
#     srcs = ["gemm_specification.proto"],
#     deps = [
#         ":gemm_backend_config_proto",
#         ":cutlass_config_proto",
#     ]
# )

# proto_library(
#     name = "conv_specification_proto",
#     srcs = ["conv_specification.proto"],
#     deps = [
#         ":conv_backend_config_proto",
#         ":cutlass_config_proto",
#     ]
# )

# proto_library(
#     name = "dim3_proto",
#     srcs = ["dim3.proto"],
# )

# proto_library(
#     name = "cuda_device_context_proto",
#     srcs = ["cuda_device_context.proto"],
# )

# proto_library(
#     name = "cuda_runtime_context_proto",
#     srcs = ["cuda_runtime_context.proto"],
#     deps = [
#         ":dim3_proto"
#     ]
# )

# proto_library(
#     name = "cuda_context_proto",
#     srcs = ["cuda_context.proto"],
#     deps = [
#         ":cuda_device_context_proto",
#         ":cuda_runtime_context_proto"
#     ]
# )

# cc_proto_library(
#     name = "arch_proto_cc",
#     deps = [":arch_proto"]
# )
alias(
    name = "arch_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:arch_proto_cc",
)

alias(
    name = "gemm_shape_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:gemm_shape_proto_cc",
)

alias(
    name = "cutlass_config_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cutlass_config_proto_cc",
)

alias(
    name = "gemm_backend_config_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:gemm_backend_config_proto_cc",
)

alias(
    name = "conv_backend_config_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:conv_backend_config_proto_cc",
)

alias(
    name = "cluster_elewise_specification_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cluster_elewise_specification_proto_cc",
)

alias(
    name = "cluster_reduce_specification_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cluster_reduce_specification_proto_cc",
)

alias(
    name = "gemm_specification_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:gemm_specification_proto_cc",
)

alias(
    name = "conv_specification_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:conv_specification_proto_cc",
)

alias(
    name = "dim3_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:dim3_proto_cc",
)

alias(
    name = "cuda_device_context_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cuda_device_context_proto_cc",
)

alias(
    name = "cuda_runtime_context_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cuda_runtime_context_proto_cc",
)

alias(
    name = "cuda_context_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:cuda_context_proto_cc",
)

alias(
    name = "graph_specification_proto_cc",
    actual = "@org_tensorflow//tensorflow/mononn_extra/proto:graph_specification_proto_cc",
)


# tf_proto_library(
#     name = "arch_proto",
#     srcs = ["arch.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "gemm_shape_proto",
#     srcs = ["gemm_shape.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "cutlass_config_proto",
#     srcs = ["cutlass_config.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         ":gemm_shape_proto",
#         ":arch_proto",
#     ],
# )

# tf_proto_library(
#     name = "gemm_backend_config_proto",
#     srcs = ["gemm_backend_config.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "conv_backend_config_proto",
#     srcs = ["conv_backend_config.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "cluster_elewise_specification_proto",
#     srcs = ["cluster_elewise_specification.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "cluster_reduce_specification_proto",
#     srcs = ["cluster_reduce_specification.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "gemm_specification_proto",
#     srcs = ["gemm_specification.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         "//mononn_engine/proto:gemm_backend_config_proto",
#         "//mononn_engine/proto:cutlass_config_proto",
#     ],
# )

# tf_proto_library(
#     name = "conv_specification_proto",
#     srcs = ["conv_specification.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         "//mononn_engine/proto:conv_backend_config_proto",
#         "//mononn_engine/proto:cutlass_config_proto",
#     ],
# )

# tf_proto_library(
#     name = "dim3_proto",
#     srcs = ["dim3.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "cuda_device_context_proto",
#     srcs = ["cuda_device_context.proto"],
#     cc_api_version = 2,
# )

# tf_proto_library(
#     name = "cuda_runtime_context_proto",
#     srcs = ["cuda_runtime_context.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         ":dim3_proto",
#     ],
# )

# tf_proto_library(
#     name = "cuda_context_proto",
#     srcs = ["cuda_context.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         ":cuda_device_context_proto",
#         ":cuda_runtime_context_proto",
#     ],
# )

# tf_proto_library(
#     name = "graph_specification_proto",
#     srcs = ["graph_specification.proto"],
#     cc_api_version = 2,
#     protodeps = [
#         "//mononn_engine/proto:conv_specification_proto",
#         "//mononn_engine/proto:gemm_specification_proto",
#         "//mononn_engine/proto:cluster_elewise_specification_proto",
#         "//mononn_engine/proto:cluster_reduce_specification_proto",
#         "//mononn_engine/proto:cuda_context_proto",
#     ],
# )
