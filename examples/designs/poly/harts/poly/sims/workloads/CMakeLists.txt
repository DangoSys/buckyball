# The workloads of the poly hart simulation: the poly device entry's, with
# the poly hart's own programs in place of the device's.
include("${DESIGN_DEVICE_BUILD}/device.cmake")
add_custom_target(design-ctests-build)
set(CTEST_GROUP component-ctests-build)
add_ball_ctests()

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
foreach(CTEST_PROFILE IN LISTS DESIGN_CPU_TARGETS)
    set(COMPONENT_WORKLOADS
        "${DESIGN_TARGET_WORKLOADS_${CTEST_PROFILE}}"
    )
    set(CTEST_GROUP component-ctests-build)
    if(IS_DIRECTORY "${COMPONENT_WORKLOADS}/ctests")
        add_subdirectory(
            "${COMPONENT_WORKLOADS}/ctests"
            "${CMAKE_CURRENT_BINARY_DIR}/prefab/${CTEST_PROFILE}/ctests"
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
foreach(CPU_TARGET IN LISTS DESIGN_CPU_TARGETS)
    set(MLIR_TEST_TARGET "${CPU_TARGET}")
    foreach(BALL IN LISTS DESIGN_PROFILES_${CPU_TARGET}_THREAD_BALLDOMAIN_BALLS)
        set(BALL_MLIR_TESTS
            "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_${CPU_TARGET}_THREAD_BALLDOMAIN_BALLS_${BALL}_SOURCE_PATH}/workloads/mlir_tests"
        )
        if(IS_DIRECTORY "${BALL_MLIR_TESTS}")
            add_subdirectory(
                "${BALL_MLIR_TESTS}"
                "${CMAKE_CURRENT_BINARY_DIR}/balls/${CPU_TARGET}/${BALL}/mlir_tests"
            )
        endif()
    endforeach()
endforeach()
add_dependencies(MLIRTest-all base-mlirtest-build balls-mlir-tests-build)

# The poly hart's own programs.
add_custom_target(poly-ctests-build)
set(CTEST_GROUP poly-ctests-build)
set(CTEST_PROFILE attention)
# Every hart boots through start.S of the C API.
set(POLY_BOOT_SOURCE "$<TARGET_OBJECTS:bbsw-start-baremetal>")
set(BUCKYBALL_DESIGN_COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/thread/ctests)
# The one hart runs Ant threads.
list(APPEND DESIGN_DEFS POLY_THREAD_HARTS=1)
list(APPEND DESIGN_DEFINES -DPOLY_THREAD_HARTS=1)
# The thread and kernel tests of the hart; its compiler tests run the device's
# reduce-unpack hart as control hart, so they stay with the device.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/thread" harts/poly/thread)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/kernel" harts/poly/kernel)
add_dependencies(design-ctests-build poly-ctests-build)
add_custom_target(design-workloads-build ALL
  DEPENDS
    design-ctests-build
    mlirtest-workloads-build
    soctests-build
  VERBATIM)

if(DEFINED DESIGN_HART_OPT_poly AND NOT (DEFINED DESIGN_TARGET_OPT_attention AND DEFINED DESIGN_TARGET_OPT_ffn))
    add_subdirectory(
        "${DESIGN_REPO}/examples/designs/poly/harts/poly/workloads/compiler/native"
        "${CMAKE_CURRENT_BINARY_DIR}/harts/poly/compiler/native"
    )
endif()

# Each hardware thread's own compiler tests, from its stack's folder.
if(IS_DIRECTORY "${DESIGN_THREAD_STACK_attention}/../workloads/compiler")
    add_subdirectory(
        "${DESIGN_THREAD_STACK_attention}/../workloads/compiler"
        "${CMAKE_CURRENT_BINARY_DIR}/threads/attention/compiler"
    )
endif()

if(IS_DIRECTORY "${DESIGN_THREAD_STACK_ffn}/../workloads/compiler")
    add_subdirectory(
        "${DESIGN_THREAD_STACK_ffn}/../workloads/compiler"
        "${CMAKE_CURRENT_BINARY_DIR}/threads/ffn/compiler"
    )
endif()

if(DEFINED DESIGN_THREAD_STACK_attention AND DEFINED DESIGN_THREAD_STACK_ffn)
    add_subdirectory(
        "${DESIGN_REPO}/examples/designs/poly/harts/poly/workloads/compiler/mxint5"
        "${CMAKE_CURRENT_BINARY_DIR}/harts/poly/compiler/mxint5"
    )
endif()
