load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_library")


cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    deps = ["@org_tensorflow//tensorflow/core/platform:logging"],
)

cc_library(
    name = "string_helpers",
    srcs = ["string_helpers.cc"],
    hdrs = ["string_helpers.h"],
    deps = [":macros"],
)

cc_library(
    name = "json",
    hdrs = ["json.hpp"],
)

cc_library(
    name = "path",
    srcs = ["path.h"],
)

cc_library(
    name = "directory",
    srcs = ["directory.cc"],
    hdrs = ["directory.h"],
    linkopts = [
        "-lstdc++fs",
    ],
    deps = [
        ":macros",
        ":path",
        ":uuid",
    ],
)

cc_library(
    name = "env_variable",
    srcs = ["env_variable.cc"],
    hdrs = ["env_variable.h"],
)

cc_library(
    name = "file",
    srcs = ["file.cc"],
    hdrs = ["file.h"],
    deps = [
        ":macros",
        ":uuid",
    ],
)

cc_library(
    name = "subprocess",
    srcs = ["subprocess.cc"],
    hdrs = ["subprocess.h"],
    deps = [
        ":macros",
    ],
)

cc_library(
    name = "uuid",
    srcs = ["uuid.cc"],
    hdrs = ["uuid.h"],
    linkopts = [
        "-luuid",
    ],
)

cc_library(
    name = "transform",
    srcs = ["transform.cc"],
    hdrs = ["transform.h"],
)

cc_library(
    name = "stl_helpers",
    srcs = ["stl_helpers.cc"],
    hdrs = ["stl_helpers.h"],
    deps = [],
)

cc_library(
    name = "protobuf",
    srcs = ["protobuf.cc"],
    hdrs = ["protobuf.h"],
    deps = [
        "//mononn_engine/helpers:macros",
        "//mononn_engine/proto:graph_specification_proto_cc",
    ],
)

cc_library(
    name = "helpers",
    hdrs = ["helpers.h"],
    deps = [
        ":directory",
        ":env_variable",
        ":file",
        ":json",
        ":macros",
        ":protobuf",
        ":stl_helpers",
        ":string_helpers",
        ":subprocess",
        ":transform",
        ":uuid",
    ],
)
