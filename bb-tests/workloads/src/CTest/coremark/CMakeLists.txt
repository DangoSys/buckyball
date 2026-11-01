set(CTEST_COREMARK_WORKLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(COREMARK_PORT_DIR ${CTEST_COREMARK_WORKLOAD_DIR}/buckyball)
set(COREMARK_SRC_DIR ${CTEST_COREMARK_WORKLOAD_DIR}/src)
set(COREMARK_THREAD ${BUCKYBALL_CONTROL_TARGET})

set(COREMARK_SOURCES
    ${COREMARK_SRC_DIR}/core_list_join.c
    ${COREMARK_SRC_DIR}/core_main.c
    ${COREMARK_SRC_DIR}/core_matrix.c
    ${COREMARK_SRC_DIR}/core_state.c
    ${COREMARK_SRC_DIR}/core_util.c
    ${COREMARK_PORT_DIR}/core_portme.c
    ${COREMARK_PORT_DIR}/main.c
)

set(COREMARK_HEADERS
    ${COREMARK_SRC_DIR}/coremark.h
    ${COREMARK_PORT_DIR}/core_portme.h
)

set(COREMARK_C_FLAGS
    -g
    -fno-common
    -O2
    -static
    -std=gnu99
    -march=rv64gc
    -mcmodel=medany
    -fno-builtin-printf
    -specs=nano.specs
    -specs=nosys.specs
    -nostartfiles
    -I${COREMARK_SRC_DIR}
    -I${COREMARK_PORT_DIR}
    -ffunction-sections
    -fdata-sections
    -Wl,--gc-sections
    -DITERATIONS=10
    -DBUCKYBALL_WORKLOAD=1
)

set(COREMARK_NAME
    "${DESIGN_NAME}-${COREMARK_THREAD}-ctest-benchmark-coremark"
)
set(EXECUTABLE "${COREMARK_NAME}-baremetal")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMAND
        ${ELF_CC} ${COREMARK_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        -o ${EXECUTABLE} $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal> ${COREMARK_SOURCES} -lm
    DEPENDS
        ${COREMARK_SOURCES}
        ${COREMARK_HEADERS}
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMENT "Building coremark: ${EXECUTABLE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
)

add_custom_target(
    coremark-all
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    COMMENT "Building coremark for Buckyball"
)
