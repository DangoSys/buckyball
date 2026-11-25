set(COHERENT_ELF ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-design-coherent_exchange-multicore-baremetal)
add_custom_command(OUTPUT ${COHERENT_ELF}
  COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
    -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
    $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
    ${DESIGN_DEFINES}
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    $<TARGET_OBJECTS:bbsw-start-baremetal> ${CMAKE_CURRENT_SOURCE_DIR}/coherent_exchange.c
    -lgcc -o ${COHERENT_ELF}
  DEPENDS coherent_exchange.c $<TARGET_OBJECTS:bbsw-start-baremetal>
    ${BUCKYBALL_BBSW_HEADERS}
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target(coherent-exchange-build DEPENDS ${COHERENT_ELF})
add_dependencies(design-ctests-build coherent-exchange-build)
add_dependencies(soctests-build coherent-exchange-build)
