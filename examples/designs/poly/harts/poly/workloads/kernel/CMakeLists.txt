foreach(
    POLY_ANT_CASE
    shared
    bank
    bank_rows
    parallel
    highpa
)
    set(POLY_ANT_THREAD attention)
    set(POLY_ANT_HEADER ant_${POLY_ANT_CASE}.h)
    set(POLY_ANT_HOST_DEFINITIONS)
    if(POLY_ANT_CASE STREQUAL "bank" OR POLY_ANT_CASE STREQUAL "bank_rows")
        set(POLY_ANT_THREAD ffn)
    endif()
    set(POLY_ANT_NAME
        "${DESIGN_NAME}-${POLY_ANT_THREAD}-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(POLY_ANT_IMAGE ${POLY_ANT_IMAGE_DIR}/${POLY_ANT_NAME}-ant)
    set(POLY_ANT_CONTROL_SOURCES)
    if(
        POLY_ANT_CASE STREQUAL "bank"
        OR POLY_ANT_CASE STREQUAL "bank_rows"
        OR POLY_ANT_CASE STREQUAL "parallel"
        OR POLY_ANT_CASE STREQUAL "shared"
    )
        list(
            APPEND POLY_ANT_CONTROL_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/ant_${POLY_ANT_CASE}_exercise.c
        )
    endif()
    foreach(SOURCE IN LISTS POLY_ANT_CONTROL_SOURCES)
        ctest_line_limit("${SOURCE}")
    endforeach()

    set(POLY_ANT_EMBED ${CMAKE_CURRENT_BINARY_DIR}/ant_${POLY_ANT_CASE}_image.S)
    enable_language(ASM)
    file(
        GENERATE OUTPUT ${POLY_ANT_EMBED}
        CONTENT
            ".section .rodata.ant,\"a\",@progbits\n.balign 8\n.global ant_image, ant_image_end\nant_image:\n.incbin \"${POLY_ANT_IMAGE}.bin\"\nant_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
    )
    set_source_files_properties(
        ${POLY_ANT_EMBED}
        PROPERTIES OBJECT_DEPENDS "${POLY_ANT_IMAGE}.bin"
    )
    set(POLY_ANT_MAIN ant_${POLY_ANT_CASE}_main.c)

    ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}")
    set(POLY_ANT_HOST_NAME
        "${DESIGN_NAME}-main-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(CTEST_PROFILE ${POLY_ANT_THREAD})
    if(NOT POLY_ANT_CASE STREQUAL "highpa")
        add_linux_ctest(${POLY_ANT_HOST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${POLY_ANT_MAIN})
        target_sources(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE ${POLY_ANT_EMBED} ${POLY_ANT_CONTROL_SOURCES}
        )
        target_compile_definitions(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE ${DESIGN_DEFS}
        )
        target_link_libraries(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE runtime-linux
        )
        add_dependencies(${POLY_ANT_HOST_NAME}-linux ${POLY_ANT_NAME}-ant-build)
        add_dependencies(poly-ctests-build ${POLY_ANT_HOST_NAME}-linux-build)
    endif()

    set(POLY_ANT_HOST_THREAD ${POLY_ANT_THREAD})
    set(POLY_ANT_CONTROL_OBJECTS)

    set(POLY_ANT_HOST_ELF ${POLY_ANT_HOST_NAME}-multicore-baremetal)
    set(POLY_ANT_HOST_DEFINES ${DESIGN_TARGET_DEFS_${POLY_ANT_HOST_THREAD}})
    list(TRANSFORM POLY_ANT_HOST_DEFINES PREPEND "-D")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_HOST_ELF}
        COMMAND
            ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
            -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
            $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> ${POLY_ANT_HOST_DEFINITIONS}
            -I${POLY_ANT_SOURCE_DIR}
            ${POLY_ANT_HOST_DEFINES}
            ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${POLY_ANT_HOST_THREAD}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            -I${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_main_THREAD_BALLDOMAIN_BALLS_f32add_SOURCE_PATH}/workloads
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
            ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}
            ${POLY_ANT_CONTROL_SOURCES} ${POLY_ANT_CONTROL_OBJECTS}
            ${POLY_ANT_EMBED} ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
            $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o
            ${POLY_ANT_HOST_ELF}
        DEPENDS
            ${POLY_ANT_MAIN}
            ${POLY_ANT_SOURCE_DIR}/${POLY_ANT_HEADER}
            ${POLY_BOOT_SOURCE}
            ${POLY_ANT_EMBED}
            $<TARGET_OBJECTS:bbsw-start-baremetal>
            ${POLY_ANT_CONTROL_SOURCES}
            ${POLY_ANT_CONTROL_OBJECTS}
            ${POLY_ANT_IMAGE}.bin
            ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
            ${BUCKYBALL_REPO_ROOT}/stack/kernel/include/thread.h
            $<TARGET_FILE:bbsw-hart-baremetal>
            ${BUCKYBALL_BBSW_HEADERS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND_EXPAND_LISTS
        VERBATIM
    )
    add_custom_target(
        ${POLY_ANT_HOST_NAME}-ctest-build
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_HOST_ELF}
    )
    add_dependencies(poly-ctests-build ${POLY_ANT_HOST_NAME}-ctest-build)
endforeach()
add_subdirectory(mlir)
add_subdirectory(ddr)

set(KERNEL_DIR ${BUCKYBALL_REPO_ROOT}/stack/kernel)
set(DATAFLOW_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM DATAFLOW_DEFINES PREPEND "-D")
set(DATAFLOW_FFN_DEFINES ${DESIGN_TARGET_DEFS_ffn})
list(TRANSFORM DATAFLOW_FFN_DEFINES PREPEND "-D")
add_custom_command(
    OUTPUT copy.elf copy.bin
    COMMAND ${ELF_CC} -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${KERNEL_DIR}/program/code.ld
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" ${DATAFLOW_DEFINES}
        ${KERNEL_DIR}/program/start.S ${CMAKE_CURRENT_SOURCE_DIR}/copy.c -o copy.elf
    COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text copy.elf copy.bin
    DEPENDS copy.c copy.h ${KERNEL_DIR}/program/start.S ${KERNEL_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_command(
    OUTPUT copy_attention.o copy_ffn.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -I${KERNEL_DIR}/include --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAFLOW_DEFINES}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c -o copy_attention.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -I${KERNEL_DIR}/include --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAFLOW_FFN_DEFINES}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c -o copy_ffn.o
    DEPENDS copy_binding.c copy.h copy.bin
    VERBATIM)
set(DATAFLOW_NAME "${DESIGN_NAME}-main-ctest-design-dataflow-multicore-baremetal")
add_custom_command(
    OUTPUT ${DATAFLOW_NAME}
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/dataflow.c
        copy_attention.o copy_ffn.o $<TARGET_FILE:kernel> -lgcc -o ${DATAFLOW_NAME}
    DEPENDS dataflow.c copy.h copy_attention.o copy_ffn.o kernel
        ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_target(poly-dataflow-build DEPENDS ${DATAFLOW_NAME})
add_dependencies(poly-ctests-build poly-dataflow-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/physical_threads.cpp")
set(PHYSICAL_THREADS_NAME "${DESIGN_NAME}-main-ctest-design-physical_threads-multicore-baremetal")
add_custom_command(
    OUTPUT ${PHYSICAL_THREADS_NAME}
    COMMAND ${ELF_CC} -std=c++20 -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES}
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${KERNEL_DIR}/include "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}"
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/physical_threads.cpp
        ${KERNEL_DIR}/src/thread.c $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o ${PHYSICAL_THREADS_NAME}
    DEPENDS physical_threads.cpp copy.h copy.bin ${POLY_BOOT_SOURCE}
        $<TARGET_OBJECTS:bbsw-start-baremetal> $<TARGET_FILE:bbsw-hart-baremetal>
        ${KERNEL_DIR}/src/thread.c ${KERNEL_DIR}/include/thread.h
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(poly-physical-threads-build DEPENDS ${PHYSICAL_THREADS_NAME})
add_dependencies(poly-ctests-build poly-physical-threads-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/dataflow_multibank.cpp")
set(MULTIBANK_NAME "${DESIGN_NAME}-main-ctest-design-dataflow_multibank-multicore-baremetal")
add_custom_command(
    OUTPUT ${MULTIBANK_NAME}
    COMMAND ${ELF_CC} -std=c++17 -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/dataflow_multibank.cpp
        copy_attention.o copy_ffn.o $<TARGET_FILE:kernel> -lgcc -o ${MULTIBANK_NAME}
    DEPENDS dataflow_multibank.cpp copy.h copy_attention.o copy_ffn.o kernel
        ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(poly-dataflow-multibank-build DEPENDS ${MULTIBANK_NAME})
add_dependencies(poly-ctests-build poly-dataflow-multibank-build)

math(EXPR HIGH_BANK_LINK_BYTES
    "${DESIGN_PROFILES_attention_HART_DATA_LINKS_0_BANKS} * ${DESIGN_PROFILES_attention_THREAD_MEM_BANK_WIDTH} * ${DESIGN_PROFILES_attention_THREAD_MEM_BANK_ENTRIES} / 8")
if(HIGH_BANK_LINK_BYTES LESS 1081360)
    message(FATAL_ERROR "dataflow_high_banks requires at least 1081360 bytes on the thread data link")
endif()
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/dataflow_high_banks.cpp")
set(HIGH_BANK_NAME "${DESIGN_NAME}-main-ctest-design-dataflow_high_banks-multicore-baremetal")
add_custom_command(
    OUTPUT ${HIGH_BANK_NAME}
    COMMAND ${ELF_CC} -std=c++17 -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/dataflow_high_banks.cpp
        copy_attention.o copy_ffn.o $<TARGET_FILE:kernel> -lgcc -o ${HIGH_BANK_NAME}
    DEPENDS dataflow_high_banks.cpp copy.h copy_attention.o copy_ffn.o kernel
        ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(poly-dataflow-high-banks-build DEPENDS ${HIGH_BANK_NAME})
add_dependencies(poly-ctests-build poly-dataflow-high-banks-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/dataflow_serial.c")
set(DATAFLOW_SERIAL_NAME "${DESIGN_NAME}-main-ctest-design-dataflow_serial-multicore-baremetal")
add_custom_command(
    OUTPUT ${DATAFLOW_SERIAL_NAME}
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${KERNEL_DIR}/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/dataflow_serial.c
        copy_attention.o copy_ffn.o $<TARGET_FILE:kernel> -lgcc -o ${DATAFLOW_SERIAL_NAME}
    DEPENDS dataflow_serial.c copy.h copy_attention.o copy_ffn.o kernel
        ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_target(poly-dataflow-serial-build DEPENDS ${DATAFLOW_SERIAL_NAME})
add_dependencies(poly-ctests-build poly-dataflow-serial-build)

set(PARTIAL_NAME "${DESIGN_NAME}-hart${DESIGN_HART_HART_poly}-ctest-dma_partial")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PARTIAL_NAME}-baremetal
    COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
        -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
        --embed-dir=${POLY_ANT_IMAGE_DIR}
        $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal,INTERFACE_LINK_OPTIONS>
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
        ${CMAKE_CURRENT_SOURCE_DIR}/dma_partial.c
        $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
        -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${PARTIAL_NAME}-baremetal
    DEPENDS dma_partial.c ${POLY_ANT_IMAGE_DIR}/dma_partial.bin
        runtime-baremetal kernel
        $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal>
        $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${PARTIAL_NAME}-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PARTIAL_NAME}-baremetal)
add_dependencies(poly-ctests-build ${PARTIAL_NAME}-build)

set(EMPTY_HART ${DESIGN_HART_HART_poly})
set(EMPTY_TEST "${DESIGN_NAME}-hart${EMPTY_HART}-ctest-kernel_empty")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/empty.c")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${EMPTY_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
    $<TARGET_PROPERTY:bbsw-crt0-hart${EMPTY_HART}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${EMPTY_HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/empty.c
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${EMPTY_TEST}-baremetal
  DEPENDS empty.c runtime-baremetal kernel
    ${DESIGN_THREAD_BUILD_attention}/loader.o ${DESIGN_THREAD_BUILD_ffn}/loader.o
    $<TARGET_OBJECTS:bbsw-crt0-hart${EMPTY_HART}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${EMPTY_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${EMPTY_TEST}-baremetal)
add_dependencies(poly-ctests-build ${EMPTY_TEST}-build)

set(BANK_PATHS_TEST "${DESIGN_NAME}-hart${DESIGN_HART_HART_poly}-ctest-bank_paths")
list(LENGTH DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_poly}_THREAD_INDICES BANK_PATH_THREADS)
if(NOT BANK_PATH_THREADS EQUAL 5)
  message(FATAL_ERROR "bank_paths exercises PolyHart's five threads; actual count=${BANK_PATH_THREADS}")
endif()
set(BANK_PATHS_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM BANK_PATHS_DEFINES PREPEND "-D")
ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/bank_paths.c")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BANK_PATHS_TEST}-baremetal
  COMMAND ${ELF_CC} ${CTEST_C_FLAGS} ${BANK_PATHS_DEFINES} -DBANK_PATH_THREADS=${BANK_PATH_THREADS}
    -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include --embed-dir=${POLY_ANT_IMAGE_DIR}
    $<TARGET_PROPERTY:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal,INTERFACE_LINK_OPTIONS>
    $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
    ${CMAKE_CURRENT_SOURCE_DIR}/bank_paths.c
    $<TARGET_FILE:runtime-baremetal> $<TARGET_FILE:kernel>
    -lm -lgcc -o ${CMAKE_CURRENT_BINARY_DIR}/${BANK_PATHS_TEST}-baremetal
  DEPENDS bank_paths.c ${POLY_ANT_IMAGE_DIR}/bank_paths.bin runtime-baremetal kernel
    $<TARGET_OBJECTS:bbsw-crt0-hart${DESIGN_HART_HART_poly}-baremetal>
    $<TARGET_OBJECTS:bbsw-syscalls-baremetal>
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(${BANK_PATHS_TEST}-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BANK_PATHS_TEST}-baremetal)
add_dependencies(poly-ctests-build ${BANK_PATHS_TEST}-build)
