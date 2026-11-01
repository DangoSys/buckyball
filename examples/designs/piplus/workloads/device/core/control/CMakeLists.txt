# The control core runs a C program on its hart.
set(HELLO "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-control_hello")
set(HELLO_ELF ${CMAKE_CURRENT_BINARY_DIR}/${HELLO}-baremetal)
add_custom_command(
    OUTPUT ${HELLO_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/hello.c -o ${HELLO_ELF}
    DEPENDS
        hello.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${HELLO}-baremetal-build DEPENDS ${HELLO_ELF})
add_dependencies(design-ctests-build ${HELLO}-baremetal-build)
