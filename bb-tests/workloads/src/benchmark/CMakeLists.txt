# ===----------------------------------------------------------------------===
# The benchmarks: each a whole suite as one program, <design>-<bench>-baremetal,
# which `--bench <bench> --os baremetal` runs, and linux/<bench>, the Linux
# program the image of `image --build '... --bench <bench>'` runs. Not part of
# the workloads:
# `bbdev bench --build '... --bench (all | <bench>)'` builds sync-bench-<bench>
# or sync-bench-all, which copy them to ${OUTPUT_BIN_DIR}/src/benchmark.
# ===----------------------------------------------------------------------===
add_subdirectory(embench)
add_subdirectory(coremark)

add_custom_target(
    sync-bench-embench
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/benchmark
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/embench/${DESIGN_NAME}-embench-baremetal
            ${OUTPUT_BIN_DIR}/src/benchmark/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/benchmark/linux
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/embench/linux/embench
            ${OUTPUT_BIN_DIR}/src/benchmark/linux/
    DEPENDS embench
    COMMENT "Syncing embench to the output directory"
)
add_custom_target(
    sync-bench-coremark
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/benchmark
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/coremark/${DESIGN_NAME}-coremark-baremetal
            ${OUTPUT_BIN_DIR}/src/benchmark/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/benchmark/linux
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/coremark/linux/coremark
            ${OUTPUT_BIN_DIR}/src/benchmark/linux/
    DEPENDS coremark
    COMMENT "Syncing coremark to the output directory"
)
add_custom_target(sync-bench-all DEPENDS sync-bench-embench sync-bench-coremark)
