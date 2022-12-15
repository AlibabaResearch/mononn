load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//mononn_engine:__subpackages__"])

cc_library(
    name = "locality_tier",
    srcs = ["locality_tier.cc"],
    hdrs = ["locality_tier.h"],
    deps = [
        "//mononn_engine/core/op:op_type",
        "//mononn_engine/helpers:string_helpers",
    ],
)

cc_library(
    name = "cluster_type",
    srcs = [
        "cluster_type.cc",
    ],
    hdrs = [
        "cluster_type.h",
    ],
)

cc_library(
    name = "op_attribute",
    srcs = [
        "op_attribute.cc",
    ],
    hdrs = [
        "op_attribute.h",
    ],
)

cc_library(
    name = "auxiliary_impl_type",
    srcs = ["auxiliary_impl_type.cc"],
    hdrs = ["auxiliary_impl_type.h"],
    deps = [],
)
