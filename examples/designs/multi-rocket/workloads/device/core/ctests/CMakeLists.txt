set(CPU_ARITHMETIC
    "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-cpu_arithmetic"
)
set(CPU_ELF ${CMAKE_CURRENT_BINARY_DIR}/${CPU_ARITHMETIC}-baremetal)
add_custom_command(
    OUTPUT ${CPU_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_arithmetic.c -o ${CPU_ELF}
    DEPENDS
        cpu_arithmetic.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${CPU_ARITHMETIC}-baremetal-build DEPENDS ${CPU_ELF})
add_dependencies(design-ctests-build ${CPU_ARITHMETIC}-baremetal-build)
add_executable(${CPU_ARITHMETIC}-linux cpu_arithmetic.c)
add_dependencies(design-ctests-build ${CPU_ARITHMETIC}-linux)
