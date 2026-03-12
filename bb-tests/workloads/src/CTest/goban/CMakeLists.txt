set(ELF_CC "riscv64-unknown-elf-gcc")

#-------------------------------------------------------------------------------
# Goban: multi-core (SPMD) baremetal workloads
# All cores run the ELF simultaneously; divergence is by mhartid.
#-------------------------------------------------------------------------------
set(CTEST_GOBAN_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/goban)
set(GOBAN_LD ${CTEST_GOBAN_WORKLOAD_DIR}/goban.ld)

set(GOBAN_C_FLAGS
  -g -fno-common -O2 -static
  -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${GOBAN_LD}
  -I${WORKLOAD_LIB_DIR}
  -I${CTEST_GOBAN_WORKLOAD_DIR}
)

# Generate a multicore baremetal ELF.
# start.S (goban version) launches all harts into main().
function(add_goban_test_target TEST_NAME SOURCE_FILE)
  set(EXECUTABLE "${TEST_NAME}_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${GOBAN_C_FLAGS}
      -o ${EXECUTABLE}
      ${CTEST_GOBAN_WORKLOAD_DIR}/start.S
      ${CTEST_GOBAN_WORKLOAD_DIR}/${SOURCE_FILE}
    DEPENDS
      ${CTEST_GOBAN_WORKLOAD_DIR}/${SOURCE_FILE}
      ${CTEST_GOBAN_WORKLOAD_DIR}/start.S
    COMMENT "Building goban multicore baremetal: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(${TEST_NAME}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building ${TEST_NAME}"
  )
endfunction()

#-------------------------------------------------------------------------------
# Build list
#-------------------------------------------------------------------------------
add_goban_test_target(ctest_goban_barrier_test      barrier_test.c)
add_goban_test_target(ctest_goban_barrier_mvin_test barrier_mvin_test.c)

# Master build target
add_custom_target(buckyball-goban-CTest-build ALL
  DEPENDS
    ctest_goban_barrier_test
    ctest_goban_barrier_mvin_test
  COMMENT "Building all Goban workloads"
  VERBATIM
)
