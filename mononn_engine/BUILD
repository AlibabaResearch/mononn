load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_binary", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//mononn_engine:__subpackages__"])


tf_cc_binary(
    name = "cpu_pressure",
    srcs = ["cpu_pressure.cc"],
    deps = [
        "//mononn_engine/tuning/profiler:thread_pool",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)