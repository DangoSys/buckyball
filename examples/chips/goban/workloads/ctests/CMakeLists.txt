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

get_filename_component(_BUCKYBALL_REPO_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

set(GOBAN_CTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GOBAN_LD ${GOBAN_CTEST_DIR}/goban.ld)
set(GOBAN_MEM_RUNTIME ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.c)

set(GOBAN_C_FLAGS
  -g -fno-common -O2 -static
  -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -DBANK_NUM=64
  -Wl,-T,${GOBAN_LD}
  -I${WORKLOAD_LIB_DIR}
  -I${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}
  -I${GOBAN_CTEST_DIR}
)

function(add_goban_test_target STEM SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before goban ctests")
  endif()
  set(TEST_NAME "${BUCKYBALL_WORKLOAD_CHIP}_${STEM}")
  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  set(EXECUTABLE "${TEST_NAME}-multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${GOBAN_C_FLAGS}
      ${ARGN}
      -o ${EXECUTABLE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_MEM_RUNTIME}
    DEPENDS
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/goban.h
      ${GOBAN_CTEST_DIR}/scu.h
      ${GOBAN_MEM_RUNTIME}
      ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
      ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h
    COMMENT "Building goban multicore baremetal: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building ${TEST_NAME}"
  )
endfunction()

function(add_goban_64t4c_test_target STEM SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before goban ctests")
  endif()
  set(TEST_NAME "${BUCKYBALL_WORKLOAD_CHIP}_${STEM}")
  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  set(EXECUTABLE "${TEST_NAME}-multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${GOBAN_C_FLAGS}
      -DGOBAN_MAX_TILES=64
      -DGOBAN_HIDDEN_HART_BASE=64
      -DGOBAN_STACK_TOP=0x80400000
      -o ${EXECUTABLE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_MEM_RUNTIME}
    DEPENDS
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/goban.h
      ${GOBAN_CTEST_DIR}/scu.h
      ${GOBAN_MEM_RUNTIME}
      ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
      ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h
    COMMENT "Building goban 64t4c baremetal: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building ${TEST_NAME}"
  )
endfunction()

function(add_goban_2t4c_test_target STEM SOURCE_FILE)
  if(BUCKYBALL_WORKLOAD_CHIP STREQUAL "")
    message(FATAL_ERROR "BUCKYBALL_WORKLOAD_CHIP must be set before goban ctests")
  endif()
  set(TEST_NAME "${BUCKYBALL_WORKLOAD_CHIP}_${STEM}")
  buckyball_enforce_ctest_line_limit(${SOURCE_FILE})
  set(EXECUTABLE "${TEST_NAME}-multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${GOBAN_C_FLAGS}
      -DGOBAN_MAX_TILES=2
      -DGOBAN_HIDDEN_HART_BASE=2
      -DNTILES=2
      -DHIDDEN_HART_BASE=2
      -o ${EXECUTABLE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_MEM_RUNTIME}
    DEPENDS
      ${GOBAN_CTEST_DIR}/${SOURCE_FILE}
      ${GOBAN_CTEST_DIR}/start.S
      ${GOBAN_CTEST_DIR}/goban.h
      ${GOBAN_CTEST_DIR}/scu.h
      ${GOBAN_MEM_RUNTIME}
      ${WORKLOAD_LIB_DIR}/bbhw/mem/mem.h
      ${BUCKYBALL_ISA_DIR}/${BUCKYBALL_CTEST_TARGET}/ballISA.h
    COMMENT "Building goban 2t4c baremetal: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building ${TEST_NAME}"
  )
endfunction()

add_goban_test_target(barrier_test barrier_test.c)
add_goban_test_target(barrier_mvin_test barrier_mvin_test.c)
add_goban_test_target(shared_barrier_mvin_test shared_barrier_mvin_test.c)
add_goban_test_target(shared_same_vbank_test shared_same_vbank_test.c)
add_goban_test_target(shared_multigroup_mvin_test shared_multigroup_mvin_test.c)
add_goban_test_target(shared_duplicate_alloc_assert_test shared_duplicate_alloc_assert_test.c)
add_goban_test_target(shared_invalid_release_assert_test shared_invalid_release_assert_test.c)
add_goban_test_target(shared_barrier_mvin_2core_test shared_barrier_mvin_test.c -DNCORES=2 -DGOBAN_CORES_PER_TILE=2 -DGOBAN_MAX_TILES=1)
add_goban_test_target(shared_same_vbank_2core_test shared_same_vbank_test.c -DNCORES=2 -DGOBAN_CORES_PER_TILE=2 -DGOBAN_MAX_TILES=1)
add_goban_test_target(shared_multigroup_mvin_2core_test shared_multigroup_mvin_test.c -DNCORES=2 -DGOBAN_CORES_PER_TILE=2 -DGOBAN_MAX_TILES=1)
add_goban_test_target(shared_barrier_mvin_8core_test shared_barrier_mvin_test.c -DNCORES=8 -DGOBAN_CORES_PER_TILE=8 -DGOBAN_MAX_TILES=1)
add_goban_test_target(shared_same_vbank_8core_test shared_same_vbank_test.c -DNCORES=8 -DGOBAN_CORES_PER_TILE=8 -DGOBAN_MAX_TILES=1)
add_goban_test_target(shared_multigroup_mvin_8core_test shared_multigroup_mvin_test.c -DNCORES=8 -DGOBAN_CORES_PER_TILE=8 -DGOBAN_MAX_TILES=1)
add_goban_2t4c_test_target(hetero_2t4c_smoke_test hetero_64t4c_smoke_test.c)
add_goban_64t4c_test_target(hetero_64t4c_smoke_test hetero_64t4c_smoke_test.c)
add_goban_2t4c_test_target(uart_rx_echo_2t4c_test uart_rx_echo_test.c)
add_goban_64t4c_test_target(uart_rx_echo_64t4c_test uart_rx_echo_test.c)

add_custom_target(goban-ctests-build ALL
  DEPENDS
    ${BUCKYBALL_WORKLOAD_CHIP}_barrier_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_barrier_mvin_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_barrier_mvin_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_same_vbank_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_multigroup_mvin_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_barrier_mvin_2core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_same_vbank_2core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_multigroup_mvin_2core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_barrier_mvin_8core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_same_vbank_8core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_shared_multigroup_mvin_8core_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_hetero_2t4c_smoke_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_hetero_64t4c_smoke_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_uart_rx_echo_2t4c_test-ctest-build
    ${BUCKYBALL_WORKLOAD_CHIP}_uart_rx_echo_64t4c_test-ctest-build
  COMMENT "Building all Goban CTest workloads"
  VERBATIM
)
