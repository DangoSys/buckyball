# POLY_ANT_CASES comes from the simulation entry: CoreLink and coherence cases
# need a power-of-two count (>= 2) of thread-bearing cores.
foreach(POLY_ANT_CASE IN LISTS POLY_ANT_CASES)
    set(POLY_ANT_THREAD attention)
    set(POLY_ANT_HEADER ant_${POLY_ANT_CASE}.h)
    set(POLY_ANT_HOST_DEFINITIONS)
    if(POLY_ANT_CASE MATCHES "^mesh_(private|shared)$")
        set(POLY_ANT_HEADER ant_mesh.h)
        if(POLY_ANT_CASE STREQUAL "mesh_private")
            set(POLY_ANT_PAYLOAD mesh_private_bank_move_test.c)
            set(POLY_ANT_HOST_DEFINITIONS -DPOLY_MESH_PRIVATE=1)
        else()
            set(POLY_ANT_PAYLOAD mesh_shared_mem_test.c)
            set(POLY_ANT_HOST_DEFINITIONS -DPOLY_MESH_PRIVATE=0)
        endif()
    endif()

    set(POLY_ANT_NAME
        "${DESIGN_NAME}-${POLY_ANT_THREAD}-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(POLY_ANT_IMAGE ${POLY_ANT_IMAGE_DIR}/${POLY_ANT_NAME}-ant)
    set(POLY_ANT_CONTROL_SOURCES)
    if(POLY_ANT_CASE STREQUAL "corelink_reduce")
        list(
            APPEND POLY_ANT_CONTROL_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/ant_corelink_reduce_control.c
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
    if(POLY_ANT_CASE MATCHES "^mesh_(private|shared)$")
        set(POLY_ANT_MAIN ant_mesh_main.c)
    endif()

    buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_MAIN}")
    set(POLY_ANT_HOST_NAME
        "${DESIGN_NAME}-main-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(BUCKYBALL_CTEST_TARGET ${POLY_ANT_THREAD})

    set(POLY_ANT_HOST_THREAD ${POLY_ANT_THREAD})
    set(POLY_ANT_CONTROL_OBJECTS)
    if(POLY_ANT_CASE STREQUAL "corelink_reduce")
        set(POLY_ANT_HOST_THREAD main)
        set(POLY_ANT_CONTROL_OBJECTS
            ${CMAKE_CURRENT_BINARY_DIR}/ant_corelink_reduce_control.o
        )
        set(POLY_ANT_DEFINES ${DESIGN_TARGET_DEFS_${POLY_ANT_THREAD}})
        list(TRANSFORM POLY_ANT_DEFINES PREPEND "-D")
        add_custom_command(
            OUTPUT ${POLY_ANT_CONTROL_OBJECTS}
            COMMAND
                ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany
                -msmall-data-limit=0 -fno-builtin -I${POLY_ANT_SOURCE_DIR}
                ${POLY_ANT_DEFINES}
                -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
                -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
                -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -c
                ${POLY_ANT_CONTROL_SOURCES} -o ${POLY_ANT_CONTROL_OBJECTS}
            DEPENDS
                ${POLY_ANT_CONTROL_SOURCES}
                ${POLY_ANT_SOURCE_DIR}/ant_corelink_reduce.h
            VERBATIM
        )
        set(POLY_ANT_CONTROL_SOURCES)
    endif()

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

set(POLY_STREAM_NAME "${DESIGN_NAME}-main-ctest-design-stream")
buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/stream_main.c")
add_executable(${POLY_STREAM_NAME}-linux stream_main.c)
target_include_directories(
    ${POLY_STREAM_NAME}-linux
    PRIVATE
        ${RUNTIME_INCLUDE_DIR}
        ${WORKLOAD_LIB_DIR}
        ${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxmm/workloads
)
# The Ball instructions of the job go into the attention thread's stream.
target_compile_definitions(
    ${POLY_STREAM_NAME}-linux
    PRIVATE ${DESIGN_DEFS} ${DESIGN_TARGET_DEFS_attention} BUCKYBALL_STREAM
)
target_link_libraries(${POLY_STREAM_NAME}-linux PRIVATE engine runtime-linux)
add_dependencies(poly-ctests-build ${POLY_STREAM_NAME}-linux)

add_subdirectory(ctests)
