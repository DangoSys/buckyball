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
        target_include_directories(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE ${RUNTIME_INCLUDE_DIR}
        )
        target_compile_definitions(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE ${DESIGN_DEFS}
        )
        target_link_libraries(
            ${POLY_ANT_HOST_NAME}-linux
            PRIVATE engine runtime-linux pthread
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
            -Wl,-T,${BBSIM_LD} ${POLY_ANT_HOST_DEFINITIONS}
            -I${POLY_ANT_SOURCE_DIR}
            ${POLY_ANT_HOST_DEFINES}
            ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
            -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/f32add/workloads
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
            ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}
            ${POLY_ANT_CONTROL_SOURCES} ${POLY_ANT_CONTROL_OBJECTS}
            ${POLY_ANT_EMBED} ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
            ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
            ${POLY_ANT_HOST_ELF}
        DEPENDS
            ${POLY_ANT_MAIN}
            ${POLY_ANT_SOURCE_DIR}/${POLY_ANT_HEADER}
            ${POLY_BOOT_SOURCE}
            ${POLY_ANT_EMBED}
            ${BAREMETAL_DIR}/start.S
            ${BBSIM_LD}
            ${POLY_ANT_CONTROL_SOURCES}
            ${POLY_ANT_CONTROL_OBJECTS}
            ${POLY_ANT_IMAGE}.bin
            ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
            ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
            ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
            ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.h
            ${RUNTIME_INCLUDE_DIR}/bb_cores.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    add_custom_target(
        ${POLY_ANT_HOST_NAME}-ctest-build
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_HOST_ELF}
    )
    add_dependencies(poly-ctests-build ${POLY_ANT_HOST_NAME}-ctest-build)
endforeach()
add_subdirectory(mlir)
