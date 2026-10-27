# Cross-core Linux control threads share one coherent LLC.
set(POLY_CONTROL_NAME
    "${DESIGN_NAME}-main-ctest-design-controllers"
)
set(POLY_CONTROL_ELF ${POLY_CONTROL_NAME}-multicore-baremetal)
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/controllers.c")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_CONTROL_ELF}
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${BBSIM_LD}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${BAREMETAL_DIR} -o ${POLY_CONTROL_ELF}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/controllers.c -lgcc
    DEPENDS
        ${POLY_BOOT_SOURCE}
        controllers.c
        ${BAREMETAL_DIR}/start.S
        ${BBSIM_LD}
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${POLY_CONTROL_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_CONTROL_ELF}
)
add_dependencies(poly-ctests-build ${POLY_CONTROL_NAME}-ctest-build)

set(BUCKYBALL_CTEST_TARGET attention)
add_buckyball_linux_ctest(${POLY_CONTROL_NAME} ${CMAKE_CURRENT_SOURCE_DIR} controllers.c)
target_include_directories(
    ${POLY_CONTROL_NAME}-linux
    PRIVATE ${RUNTIME_INCLUDE_DIR}
)
target_compile_definitions(${POLY_CONTROL_NAME}-linux PRIVATE ${DESIGN_DEFS})
target_link_libraries(${POLY_CONTROL_NAME}-linux PRIVATE pthread)
add_dependencies(poly-ctests-build ${POLY_CONTROL_NAME}-linux-build)

add_dependencies(soctests-build ${POLY_CONTROL_NAME}-ctest-build)
