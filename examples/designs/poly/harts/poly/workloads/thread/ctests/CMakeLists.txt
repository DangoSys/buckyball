foreach(
    POLY_ANT_CASE
    shared
    bank
    bank_rows
    parallel
    highpa
    coherence
)
    set(POLY_ANT_THREAD attention)
    set(POLY_ANT_PAYLOAD ant_${POLY_ANT_CASE}.c)
    set(POLY_ANT_HEADER ant_${POLY_ANT_CASE}.h)
    set(POLY_ANT_HOST_DEFINITIONS)
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
    set(ANT_RUNTIME_DIR ${BUCKYBALL_REPO_ROOT}/stack/kernel/program)
    ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/${POLY_ANT_PAYLOAD}")
    add_custom_command(
        OUTPUT ${POLY_ANT_IMAGE}.elf ${POLY_ANT_IMAGE}.bin
        COMMAND
            ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
            -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
            -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
            -nostdlib -nostartfiles -Wl,--no-relax
            -Wl,-T,${ANT_RUNTIME_DIR}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-${POLY_ANT_THREAD}-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
            ${POLY_ANT_DEFINES}
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
            -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
            -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
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
        COMMAND_EXPAND_LISTS
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

set(PARTIAL_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM PARTIAL_DEFINES PREPEND "-D")
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dma_partial.bin
    COMMAND ${ELF_CC} -O2 -g -DNDEBUG -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${ANT_RUNTIME_DIR}/code.ld
        "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${PARTIAL_DEFINES} ${ANT_RUNTIME_DIR}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/dma_partial.c -o dma_partial.elf
    COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text dma_partial.elf dma_partial.bin
    DEPENDS dma_partial.c ${ANT_RUNTIME_DIR}/start.S ${ANT_RUNTIME_DIR}/code.ld
        ${BUCKYBALL_BALL_ISA_DEPS}
    COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(poly-dma-partial-ant-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dma_partial.bin)
add_dependencies(poly-ctests-build poly-dma-partial-ant-build)

ctest_line_limit("${CMAKE_CURRENT_SOURCE_DIR}/bank_paths.c")
math(EXPR BANK_PATH_ROWS "${DESIGN_PROFILES_attention_THREAD_MEM_BANK_ENTRIES} / 8")
foreach(THREAD IN LISTS DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_poly}_THREAD_INDICES)
  if(BANK_PATH_ROWS LESS 1 OR NOT DESIGN_THREADS_${THREAD}_MEM_BANK_WIDTH EQUAL 128
      OR DESIGN_THREADS_${THREAD}_MEM_BANK_ENTRIES LESS BANK_PATH_ROWS)
    message(FATAL_ERROR "bank_paths requires 128-bit banks and ${BANK_PATH_ROWS} rows; thread ${THREAD} has ${DESIGN_THREADS_${THREAD}_MEM_BANK_WIDTH} bits, ${DESIGN_THREADS_${THREAD}_MEM_BANK_ENTRIES} rows")
  endif()
endforeach()
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_paths.bin
  COMMAND ${ELF_CC} -O2 -g -DNDEBUG -march=rv64im -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
    -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax -Wl,-T,${ANT_RUNTIME_DIR}/code.ld
    "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
    ${PARTIAL_DEFINES} ${ANT_RUNTIME_DIR}/start.S
    ${CMAKE_CURRENT_SOURCE_DIR}/bank_paths.c -o bank_paths.elf
  COMMAND ${ELF_OBJCOPY} -O binary --only-section=.text bank_paths.elf bank_paths.bin
  DEPENDS bank_paths.c ${ANT_RUNTIME_DIR}/start.S ${ANT_RUNTIME_DIR}/code.ld
    ${BUCKYBALL_BALL_ISA_DEPS}
  COMMAND_EXPAND_LISTS VERBATIM)
add_custom_target(poly-bank-paths-ant-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/bank_paths.bin)
add_dependencies(poly-ctests-build poly-bank-paths-ant-build)
