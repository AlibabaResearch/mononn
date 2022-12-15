load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")
package(default_visibility = ["//mononn_engine:__subpackages__"])
cc_library(
    name = "computation",
    hdrs = ["computation.h"],
    srcs = ["computation.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo"
    ],
)
