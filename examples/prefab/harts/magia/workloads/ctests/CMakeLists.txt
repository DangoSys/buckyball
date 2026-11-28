file(GLOB MAGIA_HEADERS CONFIGURE_DEPENDS "${MAGIA_LIB}/include/*.h")
set(MAGIA_TESTS redmule_test_mm fsync_test_mm redmule_small_mm gemm_columns_mm)
foreach(TEST IN LISTS MAGIA_TESTS)
    set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-${TEST}-baremetal")
    set(NATIVE "${CMAKE_CURRENT_BINARY_DIR}/${TEST}")
    file(MAKE_DIRECTORY "${NATIVE}")
    add_custom_command(
        OUTPUT "${NATIVE}/mesh.elf" "${NATIVE}/mesh.bin" "${NATIVE}/mesh.o"
        COMMAND ${ELF_CC}
            -march=rv32imc_zicsr_xcvelw -mabi=ilp32 -O2 -g
            -ffreestanding -fno-builtin -fno-common -fno-stack-protector
            -ffunction-sections -fdata-sections -nostdlib -nostartfiles
            -DCV32E40P -D__cv32e40p__ -DMAGIA_MAILBOX_BASE=0x60000000
            "-I${MAGIA_LIB}/include" "-I${MAGIA_LIB}/../../stack/firmware" "-T${MAGIA_LIB}/link.ld"
            "${MAGIA_LIB}/crt0.S" "${MAGIA_LIB}/runtime.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/${TEST}.c"
            -Wl,--gc-sections -o "${NATIVE}/mesh.elf"
        COMMAND ${ELF_OBJCOPY} -O binary "${NATIVE}/mesh.elf" mesh.bin
        COMMAND ${ELF_CC} -march=rv64gc -mabi=lp64d -c
            "${CMAKE_CURRENT_SOURCE_DIR}/mesh.S" -o mesh.o
        WORKING_DIRECTORY "${NATIVE}"
        DEPENDS ${MAGIA_HEADERS} "${MAGIA_LIB}/../../stack/firmware/gemm.h" "${TEST}.c" "mesh.S" "${MAGIA_LIB}/crt0.S" "${MAGIA_LIB}/runtime.c" "${MAGIA_LIB}/link.ld"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
            $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
            "${CMAKE_CURRENT_SOURCE_DIR}/mesh.c" "${NATIVE}/mesh.o"
            -Wl,--section-start=.magia=0xcc000000
            -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
        DEPENDS "mesh.c" "${NATIVE}/mesh.o"
            $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
            $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
    add_dependencies(design-ctests-build ${NAME}-build)
endforeach()

# The host's matrix product on the mesh: the tiles run the firmware of the
# hart's stack, the host its runtime.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_service-baremetal")
set(NATIVE "${CMAKE_CURRENT_BINARY_DIR}/gemm_service")
set(MAGIA_STACK "${MAGIA_LIB}/../../stack")
file(MAKE_DIRECTORY "${NATIVE}")
add_custom_command(
    OUTPUT "${NATIVE}/mesh.elf" "${NATIVE}/mesh.bin" "${NATIVE}/mesh.o"
    COMMAND ${ELF_CC}
        -march=rv32imc_zicsr_xcvelw -mabi=ilp32 -O2 -g
        -ffreestanding -fno-builtin -fno-common -fno-stack-protector
        -ffunction-sections -fdata-sections -nostdlib -nostartfiles
        -DCV32E40P -D__cv32e40p__ -DMAGIA_MAILBOX_BASE=0x60000000
        "-I${MAGIA_LIB}/include" "-I${MAGIA_STACK}/firmware" "-T${MAGIA_LIB}/link.ld"
        "${MAGIA_LIB}/crt0.S" "${MAGIA_LIB}/runtime.c"
        "${MAGIA_STACK}/firmware/service.c"
        -Wl,--gc-sections -o "${NATIVE}/mesh.elf"
    COMMAND ${ELF_OBJCOPY} -O binary "${NATIVE}/mesh.elf" mesh.bin
    COMMAND ${ELF_CC} -march=rv64gc -mabi=lp64d -c
        "${CMAKE_CURRENT_SOURCE_DIR}/mesh.S" -o mesh.o
    WORKING_DIRECTORY "${NATIVE}"
    DEPENDS ${MAGIA_HEADERS} "${MAGIA_STACK}/firmware/gemm.h" "${MAGIA_STACK}/firmware/service.c" "mesh.S"
        "${MAGIA_LIB}/crt0.S" "${MAGIA_LIB}/runtime.c" "${MAGIA_LIB}/link.ld"
    VERBATIM
)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_service.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_service.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# The same product from a Linux program: the runtime maps L2 and copies the
# matrices of the program into the DDR the image keeps from Linux.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_service-linux")
add_custom_command(
    OUTPUT "${NATIVE}/firmware-linux.o"
    COMMAND ${CMAKE_C_COMPILER} -march=rv64gc -mabi=lp64d -c
        "${MAGIA_STACK}/firmware/linux.S" -o firmware-linux.o
    WORKING_DIRECTORY "${NATIVE}"
    DEPENDS "${MAGIA_STACK}/firmware/linux.S" "${NATIVE}/mesh.bin"
    VERBATIM
)
add_executable(${NAME} gemm_service.c "${MAGIA_STACK}/runtime/mesh/linux.c" "${NATIVE}/firmware-linux.o")
add_dependencies(design-ctests-build ${NAME})

# One row, as a step of a decoder has.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_one_row-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_one_row.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_one_row.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# Two columns a tile.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_two_columns-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_two_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_two_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# As many columns as tiles.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_one_column-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_one_column.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_one_column.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# Fewer columns than tiles.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_few_columns-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_few_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_few_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# An odd number of columns.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-gemm_odd_columns-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm_odd_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c"
        "${NATIVE}/mesh.o"
        -Wl,--section-start=.magia=0xcc000000
        -lm
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "gemm_odd_columns.c" "${MAGIA_STACK}/runtime/mesh/gemm.c" "${MAGIA_STACK}/runtime/mesh/jobs.h" "${NATIVE}/mesh.o"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)

# Loom's half-precision conversions: a program of the host alone.
set(NAME "${DESIGN_NAME}-${BUCKYBALL_CONTROL_TARGET}-ctest-half_convert-baremetal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        "${CMAKE_CURRENT_SOURCE_DIR}/half_convert.c"
        -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}"
    DEPENDS "half_convert.c"
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_WORKLOAD_CONTROL_HART}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${NAME}-build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${NAME}")
add_dependencies(design-ctests-build ${NAME}-build)
