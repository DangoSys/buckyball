# ===============================================================
# SoC tests: CPU-visible design devices (CLINT, PLIC, SCU) exercised on
# the design's boot hart, independent of any Ball ISA.
# ===============================================================
set(SOCTEST_MAX_LINES 100)
set(BUCKYBALL_SOCTEST_C_FLAGS
    -g
    -fno-common
    -O2
    -static
    -march=rv64gc
    -mcmodel=medany
    -fno-builtin-printf
    -specs=nano.specs
    -specs=nosys.specs
    -nostartfiles
)

set(BUCKYBALL_SOCTEST_TARGET ${BUCKYBALL_CONTROL_TARGET})

add_custom_target(soctests-build)

function(add_buckyball_soctest SOURCE_FILE)
    get_filename_component(_stem ${SOURCE_FILE} NAME_WE)
    set(TEST_NAME
        "${DESIGN_NAME}-${BUCKYBALL_SOCTEST_TARGET}-soctest-${_stem}"
    )
    set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
    file(STRINGS ${SOURCE} _lines)
    list(LENGTH _lines _n)
    if(_n GREATER SOCTEST_MAX_LINES)
        message(
            FATAL_ERROR
            "soctest exceeds ${SOCTEST_MAX_LINES} lines: ${SOURCE} (${_n} lines)"
        )
    endif()

    set(EXECUTABLE "${TEST_NAME}-baremetal")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
        COMMAND
            ${ELF_CC} ${BUCKYBALL_SOCTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
            -o ${EXECUTABLE} $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal> ${SOURCE}
            -DCORE_HART_ID=${DESIGN_WORKLOAD_CONTROL_HART} ${DESIGN_DEFINES}
            -DTRACE_BASE=${DESIGN_DEVICE_TRACE_BASE}ULL
            -DTRACE_STRIDE=${DESIGN_DEVICE_TRACE_STRIDE}ULL
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/trace
        DEPENDS
            ${SOURCE}
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            ${CMAKE_CURRENT_SOURCE_DIR}/soc.h
            ${BUCKYBALL_REPO_ROOT}/stack/runtime/trace/event.h
        COMMENT "Building SoC test: ${EXECUTABLE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(
        ${TEST_NAME}-baremetal-build
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EXECUTABLE}
    )
    add_dependencies(soctests-build ${TEST_NAME}-baremetal-build)
endfunction()

add_buckyball_soctest(clint_timer_interrupt.c)
add_buckyball_soctest(clint_software_interrupt.c)
add_buckyball_soctest(clint_supervisor_timer.c)
add_buckyball_soctest(time_csr_clock.c)
add_buckyball_soctest(time_csr_permissions.c)
add_buckyball_soctest(lrsc_word.c)
add_buckyball_soctest(lrsc_translated.c)
add_buckyball_soctest(plic_register_access.c)
add_buckyball_soctest(trace_events.c)

add_custom_target(
    sync-soctest-bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_BIN_DIR}/src/SocTest
    COMMAND
        rsync -a --delete --exclude=CMakeFiles "--include=*/"
        "--include=*-baremetal" "--include=*-baremetal.expect" "--exclude=*"
        ${CMAKE_CURRENT_BINARY_DIR}/ ${OUTPUT_BIN_DIR}/src/SocTest/
    DEPENDS soctests-build
    COMMENT "Syncing SoC test binaries to output directory"
)
