buckyball_ctest_name(MVOVER_NAME goban chip-mvover_rows)
set(MVOVER_ELF ${MVOVER_NAME}-multicore-baremetal)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MVOVER_ELF}
  COMMAND ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-g++
    -O2 -static -march=rv64gc -mcmodel=medany -fno-builtin-printf
    -specs=nano.specs -specs=nosys.specs -nostartfiles -Wl,-T,${BBSIM_LD}
    -I${WORKLOAD_LIB_DIR} -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
    -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore
    -I${BUCKYBALL_ISA_DIR}/chip -I${BUCKYBALL_ISA_DIR}/goban
    -o ${MVOVER_ELF} ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/mvover_rows.c
    ${WORKLOAD_LIB_DIR}/buckyball.c ${BBHW_MEM_C}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mvover_rows.c
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S ${BBHW_MEM_C}
    ${WORKLOAD_LIB_DIR}/bbhw/isa/13_mvover.c
    ${BUCKYBALL_ISA_DIR}/goban/params.h ${BUCKYBALL_ISA_DIR}/chip/topology.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${MVOVER_NAME}-ctest-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MVOVER_ELF})
add_dependencies(chip-ctests-build ${MVOVER_NAME}-ctest-build)
