set(CPU_ARITHMETIC
    "${BUCKYBALL_WORKLOAD_DESIGN}-${BUCKYBALL_CONTROL_TARGET}-ctest-cpu_arithmetic"
)
set(CPU_ELF ${CMAKE_CURRENT_BINARY_DIR}/${CPU_ARITHMETIC}-baremetal)
add_custom_command(
    OUTPUT ${CPU_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS} -I${BUCKYBALL_ISA_DIR}/control
        ${BAREMETAL_DIR}/crt0.S ${BAREMETAL_DIR}/syscalls.c
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_arithmetic.c -o ${CPU_ELF}
    DEPENDS
        cpu_arithmetic.c
        ${BAREMETAL_DIR}/crt0.S
        ${BAREMETAL_DIR}/syscalls.c
        ${BBSIM_LD}
        ${BUCKYBALL_ISA_DIR}/control/params.h
)
add_custom_target(${CPU_ARITHMETIC}-baremetal-build DEPENDS ${CPU_ELF})
add_dependencies(design-ctests-build ${CPU_ARITHMETIC}-baremetal-build)
add_executable(${CPU_ARITHMETIC}-linux cpu_arithmetic.c)
add_dependencies(design-ctests-build ${CPU_ARITHMETIC}-linux)
