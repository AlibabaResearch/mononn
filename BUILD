load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = {
        "//mononn_engine/tuning:tuner_main" : "--nocheck_visibility",
        "@org_tensorflow//tensorflow/tools/pip_package:build_pip_package": "--nocheck_visibility",
    }
)
