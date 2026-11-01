include("${DESIGN_DEVICE_BUILD}/device.cmake")

add_subdirectory(
    "${WORKLOAD_SRC_DIR}/CTest/base"
    "${CMAKE_BINARY_DIR}/src/CTest/base"
)
add_subdirectory(
    "${WORKLOAD_SRC_DIR}/CTest/tlb"
    "${CMAKE_BINARY_DIR}/src/CTest/tlb"
)
add_subdirectory(
    "${WORKLOAD_SRC_DIR}/CTest/tutorial"
    "${CMAKE_BINARY_DIR}/src/CTest/tutorial"
)
add_subdirectory(
    "${WORKLOAD_SRC_DIR}/CTest/embench"
    "${CMAKE_BINARY_DIR}/src/CTest/embench"
)
add_subdirectory(
    "${WORKLOAD_SRC_DIR}/CTest/coremark"
    "${CMAKE_BINARY_DIR}/src/CTest/coremark"
)
add_dependencies(
    ctest-workloads-build
    base-ctests-build
    tlb-ctests-build
    tutorial-ctests-build
)
add_dependencies(sync-native-bin embench-all coremark-all)

# The MLIR tests of the compiler: the harness of the stack and its base tests.
add_subdirectory(
    "${DESIGN_REPO}/bb-tests/workloads/src/MLIRTest"
    "${CMAKE_BINARY_DIR}/src/MLIRTest"
)
add_subdirectory(
    "${DESIGN_REPO}/bb-tests/workloads/src/MLIRTest/base"
    "${CMAKE_BINARY_DIR}/src/MLIRTest/mlir_tests/base"
)
add_dependencies(sync-bin sync-mlirtest-bin)
add_dependencies(MLIRTest-all base-mlirtest-build)

add_subdirectory(
    "${BUCKYBALL_REPO_ROOT}/examples/designs/piplus/workloads"
    design
)
