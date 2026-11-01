foreach(
    POLY_ANT_CASE
    mxmm
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
        buckyball_enforce_ctest_line_limit("${SOURCE}")
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

    if(POLY_ANT_CASE STREQUAL "mxmm")
        set(POLY_ANT_MAIN ant_mxmm_main.cpp)
    endif()
    buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}")
    set(POLY_ANT_HOST_NAME
        "${DESIGN_NAME}-main-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(BUCKYBALL_CTEST_TARGET ${POLY_ANT_THREAD})
    if(NOT POLY_ANT_CASE STREQUAL "highpa")
        add_buckyball_linux_ctest(${POLY_ANT_HOST_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${POLY_ANT_MAIN})
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
            ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${POLY_ANT_HOST_THREAD}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            -I${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_main_THREAD_BALLDOMAIN_BALLS_f32add_SOURCE_PATH}/workloads
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
            ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}
            ${POLY_ANT_CONTROL_SOURCES} ${POLY_ANT_CONTROL_OBJECTS}
            ${POLY_ANT_EMBED} ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
            $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
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
            ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
            ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
            $<TARGET_FILE:bbsw-core-baremetal>
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
add_subdirectory(balls)

set(ENGINE_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine)
set(DATAFLOW_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM DATAFLOW_DEFINES PREPEND "-D")
set(DATAFLOW_FFN_DEFINES ${DESIGN_TARGET_DEFS_ffn})
list(TRANSFORM DATAFLOW_FFN_DEFINES PREPEND "-D")
add_custom_command(
    OUTPUT copy.elf copy.bin
    COMMAND ${ELF_CC} -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${ENGINE_DIR}/program/code.ld
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" ${DATAFLOW_DEFINES}
        ${ENGINE_DIR}/program/start.S ${CMAKE_CURRENT_SOURCE_DIR}/copy.c -o copy.elf
    COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text copy.elf copy.bin
    DEPENDS copy.c copy.h ${ENGINE_DIR}/program/start.S ${ENGINE_DIR}/program/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_command(
    OUTPUT copy_attention.o copy_ffn.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -I${ENGINE_DIR}/include --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAFLOW_DEFINES}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c -o copy_attention.o
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -I${ENGINE_DIR}/include --embed-dir=${CMAKE_CURRENT_BINARY_DIR}
        ${DATAFLOW_FFN_DEFINES}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/copy_binding.c -o copy_ffn.o
    DEPENDS copy_binding.c copy.h copy.bin
    VERBATIM)
set(DATAFLOW_NAME "${DESIGN_NAME}-main-ctest-design-dataflow-multicore-baremetal")
add_custom_command(
    OUTPUT ${DATAFLOW_NAME}
    COMMAND ${ELF_CC} -O2 -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS>
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" -I${ENGINE_DIR}/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/dataflow.c
        copy_attention.o copy_ffn.o $<TARGET_FILE:engine> -lgcc -o ${DATAFLOW_NAME}
    DEPENDS dataflow.c copy.h copy_attention.o copy_ffn.o engine
        ${POLY_BOOT_SOURCE} $<TARGET_OBJECTS:bbsw-start-baremetal>
    COMMAND_EXPAND_LISTS
    VERBATIM)
add_custom_target(poly-dataflow-build DEPENDS ${DATAFLOW_NAME})
add_dependencies(poly-ctests-build poly-dataflow-build)
