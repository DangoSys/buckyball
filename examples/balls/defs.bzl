"""Ball product surface: source graph only. Ninja compiles."""

def ball_package():
    native.filegroup(
        name = "ctests",
        srcs = native.glob(
            ["workloads/ctests/**/*.c"],
            allow_empty = False,
        ) + native.glob(
            ["workloads/ctests/**/*.h"],
            allow_empty = True,
        ),
    )
    native.filegroup(
        name = "mlir_tests",
        srcs = native.glob(
            [
                "workloads/mlir_tests/**/*.mlir",
                "workloads/mlir_tests/**/*.cpp",
                "workloads/mlir_tests/**/*.cc",
                "workloads/mlir_tests/**/*.h",
            ],
            allow_empty = True,
        ),
    )
