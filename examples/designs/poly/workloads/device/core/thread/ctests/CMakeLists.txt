foreach(
    POLY_ANT_CASE
    mxmm
    shared
    bank
    bank_rows
    parallel
    mesh_move
    highpa
    corelink
    corelink_bank
    corelink_reduce
    coherence
    mesh_private
    mesh_shared
)
    set(POLY_ANT_THREAD attention)
    set(POLY_ANT_PAYLOAD ant_${POLY_ANT_CASE}.c)
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
    if(POLY_ANT_CASE STREQUAL "bank" OR POLY_ANT_CASE STREQUAL "bank_rows")
        set(POLY_ANT_THREAD ffn)
    endif()
    set(POLY_ANT_NAME
        "${DESIGN_NAME}-${POLY_ANT_THREAD}-ctest-design-ant_${POLY_ANT_CASE}"
    )
    set(_BANK_WIDTH ${DESIGN_PROFILES_${POLY_ANT_THREAD}_THREAD_MEM_BANK_WIDTH})
    set(_BANK_ENTRIES ${DESIGN_PROFILES_${POLY_ANT_THREAD}_THREAD_MEM_BANK_ENTRIES})
    set(_MAX_PRIVATE_VBANK ${DESIGN_PROFILES_${POLY_ANT_THREAD}_THREAD_FRONTEND_MAX_PRIVATE_VBANK})
    set(POLY_ANT_DEFINES ${DESIGN_TARGET_DEFS_${POLY_ANT_THREAD}})
    list(TRANSFORM POLY_ANT_DEFINES PREPEND "-D")
    if(
        NOT _BANK_WIDTH EQUAL 128
        OR _BANK_ENTRIES LESS 128
        OR _MAX_PRIVATE_VBANK LESS 5
    )
        message(
            FATAL_ERROR
            "Ant Mxmm needs 128-bit rows, >=128 rows/bank and private banks through 5; actual width=${_BANK_WIDTH}, rows=${_BANK_ENTRIES}, privateMax=${_MAX_PRIVATE_VBANK}"
        )
    endif()
    set(POLY_ANT_IMAGE ${CMAKE_CURRENT_BINARY_DIR}/${POLY_ANT_NAME}-ant)
    set(ANT_RUNTIME_DIR ${BUCKYBALL_REPO_ROOT}/stack/engine/program)
    buckyball_enforce_ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_PAYLOAD}")
    add_custom_command(
        OUTPUT ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin
        COMMAND
            ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
            -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
            -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
            -nostdlib -nostartfiles -Wl,--no-relax
            -Wl,-T,${ANT_RUNTIME_DIR}/code.ld -I${WORKLOAD_LIB_DIR}
            ${POLY_ANT_DEFINES}
            -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxmm/workloads
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
            ${ANT_RUNTIME_DIR}/start.S
            ${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_PAYLOAD} -o
            ${POLY_ANT_IMAGE}.elf
        COMMAND
            ${ELF_OBJCOPY} -O binary --only-section=.text ${POLY_ANT_IMAGE}.elf
            ${POLY_ANT_IMAGE}.bin
        DEPENDS
            ${POLY_ANT_PAYLOAD}
            ${POLY_ANT_HEADER}
            ${ANT_RUNTIME_DIR}/start.S
            ${ANT_RUNTIME_DIR}/code.ld
            ${BUCKYBALL_BALL_ISA_DEPS}
        VERBATIM
    )
    add_custom_target(
        ${POLY_ANT_NAME}-ant-build
        DEPENDS ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin
    )
    add_dependencies(poly-ctests-build ${POLY_ANT_NAME}-ant-build)
endforeach()

set(POLY_ANT_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
set(POLY_ANT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)
