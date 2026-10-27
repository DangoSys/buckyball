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
foreach(BUCKYBALL_COMPONENT_WORKLOAD_TARGET IN LISTS DESIGN_CPU_TARGETS)
    set(COMPONENT_WORKLOADS
        "${DESIGN_TARGET_WORKLOADS_${BUCKYBALL_COMPONENT_WORKLOAD_TARGET}}"
    )
    if(
        COMPONENT_WORKLOADS
            STREQUAL
            "${BUCKYBALL_REPO_ROOT}/examples/designs/${DESIGN_NAME}/workloads"
    )
        continue()
    endif()
    set(BUCKYBALL_CTEST_TARGET "${BUCKYBALL_COMPONENT_WORKLOAD_TARGET}")
    set(BUCKYBALL_CTEST_GROUP_TARGET component-ctests-build)
    if(IS_DIRECTORY "${COMPONENT_WORKLOADS}/ctests")
        add_subdirectory(
            "${COMPONENT_WORKLOADS}/ctests"
            "${CMAKE_CURRENT_BINARY_DIR}/prefab/${BUCKYBALL_COMPONENT_WORKLOAD_TARGET}/ctests"
        )
    endif()
endforeach()
add_dependencies(
    ctest-workloads-build
    base-ctests-build
    tlb-ctests-build
    tutorial-ctests-build
    component-ctests-build
)
add_dependencies(sync-native-bin embench-all coremark-all)

add_subdirectory(
    "${WORKLOAD_SRC_DIR}/MLIRTest/base"
    "${CMAKE_BINARY_DIR}/src/MLIRTest/mlir_tests/base"
)
foreach(BUCKYBALL_TARGET IN LISTS DESIGN_CPU_TARGETS)
    set(BUCKYBALL_MLIR_ACTIVE_TARGET "${BUCKYBALL_TARGET}")
    foreach(BUCKYBALL_BALL_DIR IN LISTS DESIGN_PROFILES_${BUCKYBALL_TARGET}_THREAD_BALLDOMAIN_BALLS)
        set(BALL_MLIR_TESTS
            "${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/${BUCKYBALL_BALL_DIR}/workloads/mlir_tests"
        )
        if(IS_DIRECTORY "${BALL_MLIR_TESTS}")
            add_subdirectory(
                "${BALL_MLIR_TESTS}"
                "${CMAKE_CURRENT_BINARY_DIR}/balls/${BUCKYBALL_TARGET}/${BUCKYBALL_BALL_DIR}"
            )
        endif()
    endforeach()
endforeach()
add_dependencies(MLIRTest-all base-mlirtest-build balls-mlir-tests-build)

# The multicore Ant cases this entry's cores can run.
set(POLY_ANT_CASES
    mesh_move
    mesh_private
    mesh_shared
)
add_subdirectory(
    "${BUCKYBALL_REPO_ROOT}/examples/designs/poly/workloads"
    design
)
