# The programs of the boom hart: self-checking C tests (exit code 0 = pass).

set(CPU_ARITHMETIC "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-cpu_arithmetic")
set(CPU_ARITHMETIC_ELF ${CMAKE_CURRENT_BINARY_DIR}/${CPU_ARITHMETIC}-baremetal)
add_custom_command(
    OUTPUT ${CPU_ARITHMETIC_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_arithmetic.c -o ${CPU_ARITHMETIC_ELF}
    DEPENDS
        cpu_arithmetic.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${CPU_ARITHMETIC}-baremetal-build DEPENDS ${CPU_ARITHMETIC_ELF})
add_dependencies(design-ctests-build ${CPU_ARITHMETIC}-baremetal-build)

set(FPU "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-fpu")
set(FPU_ELF ${CMAKE_CURRENT_BINARY_DIR}/${FPU}-baremetal)
add_custom_command(
    OUTPUT ${FPU_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/fpu.c -o ${FPU_ELF}
    DEPENDS
        fpu.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${FPU}-baremetal-build DEPENDS ${FPU_ELF})
add_dependencies(design-ctests-build ${FPU}-baremetal-build)

set(ATOMICS "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-atomics")
set(ATOMICS_ELF ${CMAKE_CURRENT_BINARY_DIR}/${ATOMICS}-baremetal)
add_custom_command(
    OUTPUT ${ATOMICS_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/atomics.c -o ${ATOMICS_ELF}
    DEPENDS
        atomics.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${ATOMICS}-baremetal-build DEPENDS ${ATOMICS_ELF})
add_dependencies(design-ctests-build ${ATOMICS}-baremetal-build)

set(MEMORY "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-memory")
set(MEMORY_ELF ${CMAKE_CURRENT_BINARY_DIR}/${MEMORY}-baremetal)
add_custom_command(
    OUTPUT ${MEMORY_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/memory.c -o ${MEMORY_ELF}
    DEPENDS
        memory.c
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
)
add_custom_target(${MEMORY}-baremetal-build DEPENDS ${MEMORY_ELF})
add_dependencies(design-ctests-build ${MEMORY}-baremetal-build)
