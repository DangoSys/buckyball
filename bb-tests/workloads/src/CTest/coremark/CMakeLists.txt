set(ELF_CC "riscv64-unknown-elf-gcc")

set(CTEST_COREMARK_WORKLOAD_DIR ${CTEST_WORKLOAD_DIR}/coremark)
set(COREMARK_PORT_DIR ${CTEST_COREMARK_WORKLOAD_DIR}/buckyball)
set(COREMARK_SRC_DIR ${CTEST_COREMARK_WORKLOAD_DIR}/src)
set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)

set(COREMARK_SOURCES
  ${COREMARK_SRC_DIR}/core_list_join.c
  ${COREMARK_SRC_DIR}/core_main.c
  ${COREMARK_SRC_DIR}/core_matrix.c
  ${COREMARK_SRC_DIR}/core_state.c
  ${COREMARK_SRC_DIR}/core_util.c
  ${COREMARK_PORT_DIR}/core_portme.c
  ${COREMARK_PORT_DIR}/main.c
)

set(COREMARK_HEADERS
  ${COREMARK_SRC_DIR}/coremark.h
  ${COREMARK_PORT_DIR}/core_portme.h
)

set(COREMARK_C_FLAGS
  -g -fno-common -O2 -static -std=gnu99 -march=rv64gc -mcmodel=medany
  -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
  -Wl,-T,${BBSIM_LD}
  -I${COREMARK_SRC_DIR}
  -I${COREMARK_PORT_DIR}
  -I${WORKLOAD_LIB_DIR}
  -ffunction-sections -fdata-sections
  -Wl,--gc-sections
  -DITERATIONS=10
  -DBUCKYBALL_WORKLOAD=1
)

function(add_coremark_singlecore_target)
  set(EXECUTABLE "coremark_singlecore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${COREMARK_C_FLAGS} -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_DIR}/test.c
      ${COREMARK_SOURCES}
      -lm
    DEPENDS ${COREMARK_SOURCES}
      ${COREMARK_HEADERS}
      ${BBSW_BAREMETAL_DIR}/crt0.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_DIR}/test.c
    COMMENT "Building coremark singlecore: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(coremark_singlecore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

function(add_coremark_multicore_target)
  set(EXECUTABLE "coremark_multicore-baremetal")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND ${ELF_CC} ${COREMARK_C_FLAGS} -o ${EXECUTABLE}
      ${BBSW_BAREMETAL_DIR}/start.S
      -DMULTICORE=3
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_DIR}/test.c
      ${COREMARK_SOURCES}
      -lm
    DEPENDS ${COREMARK_SOURCES}
      ${COREMARK_HEADERS}
      ${BBSW_BAREMETAL_DIR}/start.S
      ${BBSW_BAREMETAL_DIR}/syscalls.c
      ${BBSW_BAREMETAL_DIR}/test.c
    COMMENT "Building coremark multicore: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_target(coremark_multicore_baremetal
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE})
endfunction()

add_coremark_singlecore_target()
add_coremark_multicore_target()

add_custom_target(coremark-all DEPENDS
  coremark_singlecore_baremetal
  coremark_multicore_baremetal
  COMMENT "Building all coremark benchmarks for Buckyball"
)
