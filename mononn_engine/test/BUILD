load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary",)
load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")


tf_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:backend"
    ],
)

cc_library(
    name = "testlib",
    hdrs = ["testlib.h"],
    srcs = ["testlib.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/platform:logging",
        # "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        # "@org_tensorflow//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
        # "@org_tensorflow//tensorflow/core",
        # "@org_tensorflow//tensorflow/core:all_kernels",
        # "@org_tensorflow//tensorflow/core:direct_session",
        # "@org_tensorflow//tensorflow/core:ops",
    ]
)

