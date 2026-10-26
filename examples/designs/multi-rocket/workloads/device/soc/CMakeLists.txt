set(COHERENT_ELF ${CMAKE_CURRENT_BINARY_DIR}/${BUCKYBALL_WORKLOAD_DESIGN}-${BUCKYBALL_CONTROL_TARGET}-ctest-design-coherent_exchange-multicore-baremetal)
add_custom_command(OUTPUT ${COHERENT_ELF}
  COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
    -I${BUCKYBALL_ISA_DIR}/design
    ${BAREMETAL_DIR}/start.S ${CMAKE_CURRENT_SOURCE_DIR}/coherent_exchange.c
    -lgcc -o ${COHERENT_ELF}
  DEPENDS coherent_exchange.c ${BAREMETAL_DIR}/start.S ${BBSIM_LD}
    ${BUCKYBALL_ISA_DIR}/design/instances.h)
add_custom_target(coherent-exchange-build DEPENDS ${COHERENT_ELF})
add_dependencies(design-ctests-build coherent-exchange-build)
add_dependencies(soctests-build coherent-exchange-build)
