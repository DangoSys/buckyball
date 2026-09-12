if(NOT DEFINED ELF_CC)
  set(ELF_CC "riscv64-unknown-elf-gcc")
endif()

if(NOT DEFINED BUCKYBALL_CTEST_TARGET OR BUCKYBALL_CTEST_TARGET STREQUAL "")
  message(FATAL_ERROR "BUCKYBALL_CTEST_TARGET is required for goban ctests")
endif()
list(FIND BUCKYBALL_TARGETS "${BUCKYBALL_CTEST_TARGET}" _goban_target_index)
if(_goban_target_index EQUAL -1)
  message(FATAL_ERROR "unknown C test target: ${BUCKYBALL_CTEST_TARGET}")
endif()
if(NOT EXISTS "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h")
  message(FATAL_ERROR
    "missing target ISA header ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h")
endif()
if(NOT EXISTS "${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h")
  message(FATAL_ERROR
    "missing target params.h ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h")
endif()

set(GOBAN_CTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GOBAN_LD ${GOBAN_CTEST_DIR}/goban.ld)
set(GOBAN_MEM_RUNTIME ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)
set(GOBAN_MULTICORE_DIR ${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore)
set(GOBAN_START ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S)
set(GOBAN_C_FLAGS
  -g -fno-common -O2 -static
  -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${GOBAN_LD}
  -I${WORKLOAD_LIB_DIR}
  -I${BUCKYBALL_ISA_DIR}/chip
  -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
  -I${GOBAN_MULTICORE_DIR}
  -I${GOBAN_CTEST_DIR}
  -I${BUCKYBALL_REPO_ROOT}/examples/balls/transpose/workloads
  -I${BUCKYBALL_REPO_ROOT}/examples/balls/smatmul/workloads
  -I${BUCKYBALL_REPO_ROOT}/examples/balls/toint8/workloads
  -I${BUCKYBALL_REPO_ROOT}/examples/balls/int2fp/workloads
)

function(add_goban_test_target STEM SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before goban ctests")
  endif()
  buckyball_ctest_name(TEST_NAME ${BUCKYBALL_CTEST_TARGET} chip-${STEM})
  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  set(EXECUTABLE "${TEST_NAME}-multicore-baremetal")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${GOBAN_C_FLAGS}
      -o ${EXECUTABLE}
      ${GOBAN_START}
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_MEM_RUNTIME}
    DEPENDS
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_START}
      ${GOBAN_CTEST_DIR}/goban.h
      ${GOBAN_CTEST_DIR}/scu.h
      ${GOBAN_MEM_RUNTIME}
      ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
      ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h
      ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/params.h
      ${BUCKYBALL_ISA_DIR}/chip/topology.h
      ${BUCKYBALL_REPO_ROOT}/examples/balls/transpose/workloads/isa/transpose.h
      ${BUCKYBALL_REPO_ROOT}/examples/balls/smatmul/workloads/isa/smatmul.h
      ${BUCKYBALL_REPO_ROOT}/examples/balls/toint8/workloads/isa/quant.h
      ${BUCKYBALL_REPO_ROOT}/examples/balls/int2fp/workloads/isa/int2fp.h
    COMMENT "Building goban multicore baremetal: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building ${TEST_NAME}"
  )
  set_property(GLOBAL APPEND PROPERTY GOBAN_CTEST_BUILD_TARGETS
    ${TEST_NAME}-ctest-build)
endfunction()

add_goban_test_target(barrier_test barrier_test.c)
add_goban_test_target(barrier_mvin_test barrier_mvin_test.c)
add_goban_test_target(shared_barrier_mvin_test shared_barrier_mvin_test.c)
add_goban_test_target(shared_same_vbank_test shared_same_vbank_test.c)
add_goban_test_target(shared_multigroup_mvin_test shared_multigroup_mvin_test.c)
add_goban_test_target(shared_transpose_test shared_transpose_test.c)
add_goban_test_target(shared_smatmul_quant_int2fp_pipeline_test
  shared_smatmul_quant_int2fp_pipeline_test.c)
add_goban_test_target(topology_smoke_test topology_smoke_test.c)
add_goban_test_target(uart_rx_echo_test uart_rx_echo_test.c)

get_property(GOBAN_CTEST_BUILD_TARGETS GLOBAL PROPERTY GOBAN_CTEST_BUILD_TARGETS)
add_custom_target(goban-ctests-build ALL
  DEPENDS ${GOBAN_CTEST_BUILD_TARGETS}
  COMMENT "Building all Goban CTest workloads"
  VERBATIM
)
