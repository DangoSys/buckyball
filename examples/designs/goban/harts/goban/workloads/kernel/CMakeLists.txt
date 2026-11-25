set(KERNEL_DIR ${BUCKYBALL_REPO_ROOT}/stack/kernel)
set(MATMUL_THREAD_DEFS ${DESIGN_TARGET_DEFS_matmul})
list(TRANSFORM MATMUL_THREAD_DEFS PREPEND "-D")
set(MATMUL_NAME
    "${DESIGN_NAME}-matmul-ctest-design-matmul_threads"
)
set(MATMUL_ELF ${MATMUL_NAME}-baremetal)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${MATMUL_ELF}
    COMMAND
        ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} -I${KERNEL_DIR}/include
        -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/smatmul/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        $<TARGET_OBJECTS:bbsw-start-baremetal> ${CMAKE_CURRENT_SOURCE_DIR}/matmul_threads.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime>
        # How the device's harts run jobs and its hardware threads are loaded.
        $<TARGET_PROPERTY:bindings,INTERFACE_LINK_OPTIONS> $<TARGET_FILE:kernel>
        -lgcc -o ${MATMUL_ELF}
    DEPENDS
        matmul_threads.c
        ${KERNEL_DIR}/include/stream.h
        ${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa/bb_isa.h
        runtime-baremetal
        runtime
        kernel
        ${KERNEL_DIR}/include/thread.h
        ${KERNEL_DIR}/include/kernel.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${MATMUL_NAME}-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${MATMUL_ELF}
)
add_dependencies(design-ctests-build ${MATMUL_NAME}-ctest-build)

add_custom_command(
    OUTPUT copy.elf copy.bin
    COMMAND
        ${ELF_CC} -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${KERNEL_DIR}/program/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-matmul-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} ${KERNEL_DIR}/program/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/copy.c -o copy.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text copy.elf
        copy.bin
    DEPENDS
        copy.c copy.h
        ${KERNEL_DIR}/program/start.S ${KERNEL_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
set(DATAFLOW_NAME "${DESIGN_NAME}-matmul-ctest-design-dataflow")
set(DATAFLOW_ELF ${DATAFLOW_NAME}-baremetal)
add_custom_command(
    OUTPUT ${DATAFLOW_ELF}
    COMMAND
        ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> ${CMAKE_CURRENT_SOURCE_DIR}/dataflow.c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
        -lgcc -o ${DATAFLOW_ELF}
    DEPENDS
        dataflow.c copy_binding.c copy.h copy.bin
        runtime-baremetal kernel
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DATAFLOW_NAME}-ctest-build DEPENDS ${DATAFLOW_ELF})
add_dependencies(design-ctests-build ${DATAFLOW_NAME}-ctest-build)

# The same products under Linux: the OS owns the matmul threads and the
# kernel driver issues the management instructions (runtime-linux).
add_executable(${MATMUL_NAME}-linux matmul_threads_linux.c)
target_include_directories(
    ${MATMUL_NAME}-linux
    PRIVATE
        ${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
        ${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/smatmul/workloads
)
target_compile_definitions(
    ${MATMUL_NAME}-linux
    PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_matmul}
)
target_link_libraries(${MATMUL_NAME}-linux PRIVATE runtime-linux)
add_dependencies(design-ctests-build ${MATMUL_NAME}-linux)

set(STREAM_NAME "${DESIGN_NAME}-matmul-ctest-design-stream-linux")
add_executable(${STREAM_NAME} stream_linux.c)
target_include_directories(${STREAM_NAME} PRIVATE ${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa)
target_compile_definitions(${STREAM_NAME} PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_matmul})
target_link_libraries(${STREAM_NAME} PRIVATE runtime-linux)
add_dependencies(design-ctests-build ${STREAM_NAME})

if(NOT DESIGN_PROFILES_matmul_THREAD_MEM_BANK_WIDTH EQUAL 128)
    message(FATAL_ERROR "gp_scalar_fmul requires 128-bit bank rows, got ${DESIGN_PROFILES_matmul_THREAD_MEM_BANK_WIDTH}")
endif()
add_custom_command(
    OUTPUT gp_scalar_fmul.elf gp_scalar_fmul.bin
    COMMAND
        ${ELF_CC} -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${KERNEL_DIR}/program/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-matmul-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} ${KERNEL_DIR}/program/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/gp_scalar_fmul.c -o gp_scalar_fmul.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text gp_scalar_fmul.elf
        gp_scalar_fmul.bin
    DEPENDS
        gp_scalar_fmul.c
        ${KERNEL_DIR}/program/start.S ${KERNEL_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
set(GP_SCALAR_FMUL_NAME "${DESIGN_NAME}-matmul-ctest-design-gp_scalar_fmul")
set(GP_SCALAR_FMUL_ELF ${GP_SCALAR_FMUL_NAME}-baremetal)
add_custom_command(
    OUTPUT ${GP_SCALAR_FMUL_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib
        -I${BUCKYBALL_REPO_ROOT}/bb-tests/workloads/lib/bbhw/isa
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/gp_scalar_fmul_main.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
        -lgcc -o ${GP_SCALAR_FMUL_ELF}
    DEPENDS
        gp_scalar_fmul_main.c gp_scalar_fmul.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal kernel
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${GP_SCALAR_FMUL_NAME}-ctest-build DEPENDS ${GP_SCALAR_FMUL_ELF})
add_dependencies(design-ctests-build ${GP_SCALAR_FMUL_NAME}-ctest-build)

set(COPY_OVERLAP_NAME "${DESIGN_NAME}-matmul-ctest-design-copy_overlap")
set(COPY_OVERLAP_ELF ${COPY_OVERLAP_NAME}-baremetal)
add_custom_command(
    OUTPUT ${COPY_OVERLAP_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/copy_overlap.c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
        -lgcc -o ${COPY_OVERLAP_ELF}
    DEPENDS
        copy_overlap.c copy_binding.c copy.h copy.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal kernel
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${COPY_OVERLAP_NAME}-ctest-build DEPENDS ${COPY_OVERLAP_ELF})
add_dependencies(design-ctests-build ${COPY_OVERLAP_NAME}-ctest-build)

set(COPY_OVERLAP_SMALL_NAME "${DESIGN_NAME}-matmul-ctest-design-copy_overlap_small")
set(COPY_OVERLAP_SMALL_ELF ${COPY_OVERLAP_SMALL_NAME}-baremetal)
add_custom_command(
    OUTPUT ${COPY_OVERLAP_SMALL_ELF}
    COMMAND
        ${ELF_CC} ${CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/copy_overlap_small.c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
        -lgcc -o ${COPY_OVERLAP_SMALL_ELF}
    DEPENDS
        copy_overlap_small.c copy_binding.c copy.h copy.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal kernel
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${COPY_OVERLAP_SMALL_NAME}-ctest-build DEPENDS ${COPY_OVERLAP_SMALL_ELF})
add_dependencies(design-ctests-build ${COPY_OVERLAP_SMALL_NAME}-ctest-build)
