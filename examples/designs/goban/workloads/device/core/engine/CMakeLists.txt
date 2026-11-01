set(ENGINE_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine)
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
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} -I${ENGINE_DIR}/include
        -I${BUCKYBALL_REPO_ROOT}/stack/isa
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/smatmul/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        $<TARGET_OBJECTS:bbsw-start-baremetal> ${CMAKE_CURRENT_SOURCE_DIR}/matmul_threads.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:runtime>
        # How the device's cores run jobs and its hardware threads are loaded.
        $<TARGET_PROPERTY:bindings,INTERFACE_LINK_OPTIONS> $<TARGET_FILE:engine>
        -lgcc -o ${MATMUL_ELF}
    DEPENDS
        matmul_threads.c
        ${ENGINE_DIR}/include/stream.h
        ${BUCKYBALL_REPO_ROOT}/stack/isa/bb_isa.h
        runtime-baremetal
        runtime
        engine
        ${ENGINE_DIR}/include/thread.h
        ${ENGINE_DIR}/include/engine.h
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
        -Wl,-T,${ENGINE_DIR}/program/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-matmul-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} ${ENGINE_DIR}/program/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/copy.c -o copy.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text copy.elf
        copy.bin
    DEPENDS
        copy.c copy.h
        ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
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
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${ENGINE_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> ${CMAKE_CURRENT_SOURCE_DIR}/dataflow.c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
        -lgcc -o ${DATAFLOW_ELF}
    DEPENDS
        dataflow.c copy_binding.c copy.h copy.bin
        runtime-baremetal engine
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DATAFLOW_NAME}-ctest-build DEPENDS ${DATAFLOW_ELF})
add_dependencies(design-ctests-build ${DATAFLOW_NAME}-ctest-build)

# The same products under Linux: the OS owns the matmul threads and the
# engine driver issues the management instructions (runtime-linux).
add_executable(${MATMUL_NAME}-linux matmul_threads_linux.c)
target_include_directories(
    ${MATMUL_NAME}-linux
    PRIVATE
        ${BUCKYBALL_REPO_ROOT}/stack/isa
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
target_include_directories(${STREAM_NAME} PRIVATE ${BUCKYBALL_REPO_ROOT}/stack/isa)
target_compile_definitions(${STREAM_NAME} PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_matmul})
target_link_libraries(${STREAM_NAME} PRIVATE runtime-linux)
add_dependencies(design-ctests-build ${STREAM_NAME})

add_custom_command(
    OUTPUT ant_cycles.elf ant_cycles.bin
    COMMAND
        ${ELF_CC} -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${ENGINE_DIR}/program/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-matmul-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} ${ENGINE_DIR}/program/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ant_cycles.c -o ant_cycles.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ant_cycles.elf
        ant_cycles.bin
    DEPENDS
        ant_cycles.c
        ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
set(ANT_CYCLES_NAME "${DESIGN_NAME}-matmul-ctest-design-ant_cycles")
set(ANT_CYCLES_ELF ${ANT_CYCLES_NAME}-baremetal)
add_custom_command(
    OUTPUT ${ANT_CYCLES_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${ENGINE_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/ant_cycles_main.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
        -lgcc -o ${ANT_CYCLES_ELF}
    DEPENDS
        ant_cycles_main.c ant_cycles.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal engine
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${ANT_CYCLES_NAME}-ctest-build DEPENDS ${ANT_CYCLES_ELF})
add_dependencies(design-ctests-build ${ANT_CYCLES_NAME}-ctest-build)

add_custom_command(
    OUTPUT ball_cycles.elf ball_cycles.bin
    COMMAND
        ${ELF_CC} -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${ENGINE_DIR}/program/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-matmul-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${MATMUL_THREAD_DEFS} ${ENGINE_DIR}/program/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_cycles.c -o ball_cycles.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ball_cycles.elf
        ball_cycles.bin
    DEPENDS
        ball_cycles.c
        ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
set(BALL_CYCLES_NAME "${DESIGN_NAME}-matmul-ctest-design-ball_cycles")
set(BALL_CYCLES_ELF ${BALL_CYCLES_NAME}-baremetal)
add_custom_command(
    OUTPUT ${BALL_CYCLES_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${ENGINE_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_cycles_main.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
        -lgcc -o ${BALL_CYCLES_ELF}
    DEPENDS
        ball_cycles_main.c ball_cycles.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal engine
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${BALL_CYCLES_NAME}-ctest-build DEPENDS ${BALL_CYCLES_ELF})
add_dependencies(design-ctests-build ${BALL_CYCLES_NAME}-ctest-build)


set(COPY_OVERLAP_NAME "${DESIGN_NAME}-matmul-ctest-design-copy_overlap")
set(COPY_OVERLAP_ELF ${COPY_OVERLAP_NAME}-baremetal)
add_custom_command(
    OUTPUT ${COPY_OVERLAP_ELF}
    COMMAND
        ${ELF_CC} ${BUCKYBALL_CTEST_C_FLAGS} -msmall-data-limit=0 -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} ${MATMUL_THREAD_DEFS} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${ENGINE_DIR}/include
        --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/copy_overlap.c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:engine>
        -lgcc -o ${COPY_OVERLAP_ELF}
    DEPENDS
        copy_overlap.c copy_binding.c copy.h copy.bin $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        runtime-baremetal engine
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${COPY_OVERLAP_NAME}-ctest-build DEPENDS ${COPY_OVERLAP_ELF})
add_dependencies(design-ctests-build ${COPY_OVERLAP_NAME}-ctest-build)
