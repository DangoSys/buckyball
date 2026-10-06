set(POLY_MESH_CXX ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-elf-g++)

buckyball_ctest_name(POLY_MESH_NAME attention chip-mesh_shared_bank_roundtrip)
set(POLY_MESH_ELF ${POLY_MESH_NAME}-baremetal)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_MESH_ELF}
  COMMAND ${POLY_MESH_CXX} -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD}
    -I${WORKLOAD_LIB_DIR}
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
    -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore
    -I${BUCKYBALL_ISA_DIR}/chip
    -I${BUCKYBALL_ISA_DIR}/attention
    -o ${POLY_MESH_ELF}
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/crt0.S
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_shared_mem_test.c
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${BBHW_MEM_C}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_shared_mem_test.c
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/crt0.S
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${WORKLOAD_LIB_DIR}/buckyball.h
    ${BUCKYBALL_ISA_DIR}/attention/params.h
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/dma.h
    ${BBHW_MEM_C}
    ${BUCKYBALL_ISA_DIR}/chip/topology.h
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${POLY_MESH_NAME}-ctest-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_MESH_ELF})
add_dependencies(poly-ctests-build ${POLY_MESH_NAME}-ctest-build)

buckyball_ctest_name(POLY_MOVE_NAME attention chip-mesh_private_bank_move)
set(POLY_MOVE_ELF ${POLY_MOVE_NAME}-multicore-baremetal)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_MOVE_ELF}
  COMMAND ${POLY_MESH_CXX} -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD}
    -I${WORKLOAD_LIB_DIR}
    -I${BUCKYBALL_REPO_ROOT}/stack/runtime/include
    -I${WORKLOAD_LIB_DIR}/bbsw/baremetal/multicore
    -I${BUCKYBALL_ISA_DIR}/chip
    -I${BUCKYBALL_ISA_DIR}/attention
    -o ${POLY_MOVE_ELF}
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_private_bank_move_test.c
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${BBHW_MEM_C}
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_private_bank_move_test.c
    ${WORKLOAD_LIB_DIR}/buckyball.c
    ${WORKLOAD_LIB_DIR}/buckyball.h
    ${BUCKYBALL_ISA_DIR}/attention/params.h
    ${BUCKYBALL_REPO_ROOT}/stack/runtime/include/dma.h
    ${WORKLOAD_LIB_DIR}/bbsw/baremetal/start.S
    ${BBHW_MEM_C}
    ${BUCKYBALL_ISA_DIR}/chip/topology.h
    ${WORKLOAD_LIB_DIR}/bbhw/isa/13_mvover.c
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(${POLY_MOVE_NAME}-ctest-build
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_MOVE_ELF})
add_dependencies(poly-ctests-build ${POLY_MOVE_NAME}-ctest-build)

foreach(POLY_MXMM_CORE attention ffn)
  set(BUCKYBALL_CTEST_TARGET ${POLY_MXMM_CORE})
  add_buckyball_ctests(mxmm_fp32_small.c mxmm_mxfp8_small.c)
endforeach()
set(BUCKYBALL_CTEST_TARGET attention)
