load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cuda_library(
    name = "cupti_profiling_session",
    srcs = [
        "cupti_profiling_session.cu.cc",
    ],
    hdrs = [
        "cupti_profiling_session.h",
    ],
    deps = [
        "//mononn_engine/helpers:macros",
    ],
    linkopts = [
        "-L /usr/local/cuda/lib64/",
        "-lcupti",
        "-lcuda",
        "-lnvperf_host",
        "-lnvperf_target",
    ],
    # alwayslink = 1
)

tf_cc_binary(
    name = "cupti_test_main",
    srcs = ["cupti_test_main.cc"],
    deps = [
        ":cupti_profiling_session"
    ],
    # copts = [
    #     "-L /usr/local/cuda/lib64/",
    # ],
    # copts = [
    #     "-L/usr/local/cuda/lib64",
    #     "-lcuda",
    #     "-lcupti",
    #     "-lnvperf_host",
    #     "-lnvperf_target",
    # ],
    # linkopts = [
    #     "-lcuda",
    #     "-lcupti",
    #     "-lnvperf_host",
    #     "-lnvperf_target",
    # ]
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["subprocess.h"],
    deps = [
        "//mononn_engine/helpers",
    ],
)

tf_cc_binary(
    name = "subprocess_main",
    srcs = ["subprocess_main.cc"],
    deps = [
        ":subprocess",
    ],
)

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/platform:logging",
        "//mononn_engine/helpers:env_variable",
    ],
)

cc_library(
    name = "profiling_result",
    srcs = ["profiling_result.cc"],
    hdrs = ["profiling_result.h"],
    deps = [
        "//mononn_engine/helpers:macros"
    ],
)

cc_library(
    name = "parallel_profiling_queue",
    srcs = ["parallel_profiling_queue.cc"],
    hdrs = ["parallel_profiling_queue.h"],
    deps = [
        ":profiling_result",
        ":subprocess",
        ":thread_pool",
        "//mononn_engine/config",
        "//mononn_engine/proto:graph_specification_proto_cc",
    ],
)

cc_library(
    name = "timer",
    srcs = ["timer.cc"],
    hdrs = ["timer.h"],
    deps = [
        "@org_tensorflow//tensorflow/core/platform:logging",
    ],
)
