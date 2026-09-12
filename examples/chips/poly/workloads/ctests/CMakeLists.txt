set(POLY_VEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vecmat16_atranspose_16x16)
buckyball_ctest_name(POLY_VEC_NAME decode chip-vecmat16_atranspose_16x16)
set(POLY_VEC_ELF ${POLY_VEC_NAME}-multicore-baremetal)
set(POLY_VEC_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-g++)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_VEC_ELF}
  COMMAND ${POLY_VEC_CXX} -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD}
    -I${WORKLOAD_LIB_DIR}
    -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore
    -I${BUCKYBALL_ISA_DIR}/chip
    -I${BUCKYBALL_ISA_DIR}/decode
    -I${BUCKYBALL_REPO_ROOT}/examples/balls/vector/workloads
    -o ${POLY_VEC_ELF}
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${POLY_VEC_DIR}/vecmat16_atranspose_16x16-main.cpp
    ${POLY_VEC_DIR}/prefill.c
    ${POLY_VEC_DIR}/decode.c
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${BBHW_MEM_C}
  DEPENDS
    ${POLY_VEC_DIR}/vecmat16_atranspose_16x16-main.cpp
    ${POLY_VEC_DIR}/prefill.c
    ${POLY_VEC_DIR}/decode.c
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${BUCKYBALL_ISA_DIR}/chip/topology.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${POLY_VEC_NAME}-ctest-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_VEC_ELF})
add_dependencies(poly-ctests-build ${POLY_VEC_NAME}-ctest-build)
