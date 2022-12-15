load("@org_tensorflow//tensorflow/core/platform:rules_cc.bzl", "cc_binary", "cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//mononn_engine:__subpackages__"])


tf_cc_binary(
    name = "tf2xla_compile",
    srcs = ["tf2xla_compile.cc"],
    deps = [
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/tf2xla",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/client",
        "@org_tensorflow//tensorflow/compiler/xla/client:client_library",
        "@org_tensorflow//tensorflow/compiler/xla/client:local_client",
        "@org_tensorflow//tensorflow/compiler/xla/client:xla_computation",
        "@org_tensorflow//tensorflow/compiler/xla/service",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/service:interpreter_plugin",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_binary(
    name = "cpu_pressure",
    srcs = ["cpu_pressure.cc"],
    deps = [
        "//mononn_engine/tuning/profiler:thread_pool",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

tf_cc_binary(
    name = "statusor_test",
    srcs = ["statusor_test.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/platform:statusor",
    ],
)