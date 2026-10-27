set(COHERENT_ELF ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-design-coherent_exchange-multicore-baremetal)
add_custom_command(OUTPUT ${COHERENT_ELF}
  COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
    ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR}
    ${BAREMETAL_DIR}/start.S ${CMAKE_CURRENT_SOURCE_DIR}/coherent_exchange.c
    -lgcc -o ${COHERENT_ELF}
  DEPENDS coherent_exchange.c ${BAREMETAL_DIR}/start.S ${BBSIM_LD}
    ${RUNTIME_INCLUDE_DIR}/bb_cores.h
  VERBATIM)
add_custom_target(coherent-exchange-build DEPENDS ${COHERENT_ELF})
add_dependencies(design-ctests-build coherent-exchange-build)
add_dependencies(soctests-build coherent-exchange-build)
