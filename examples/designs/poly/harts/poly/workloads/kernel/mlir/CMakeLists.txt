# Host programs of the MLIR tests of the Balls of poly's threads: each embeds
# the thread image thread/mlir built for the test, runs it on the attention
# thread of every hart and checks what it wrote.
set(POLY_MLIR_HOST_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_MLIR_HOST_DEFINES PREPEND "-D")
set(POLY_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../thread/mlir")
enable_language(ASM)

# bank_mxint8quantize_random
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8quantize_random_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8quantize_random_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8quantize_random_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
        $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-multicore-baremetal
    DEPENDS
        bank_mxint8quantize_random_main.c
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8quantize_random_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/include/thread.h
        $<TARGET_FILE:bbsw-hart-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8quantize_random-ctest-build)

# bank_mxint8matmul_accumulate
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8matmul_accumulate_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-multicore-baremetal
    COMMAND
        ${ELF_CC} -std=c++17 -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-hart-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/kernel/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8matmul_accumulate_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8matmul_accumulate_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
        $<TARGET_FILE:bbsw-hart-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-multicore-baremetal
    DEPENDS
        bank_mxint8matmul_accumulate_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8matmul_accumulate_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/kernel/include/thread.h
        $<TARGET_FILE:bbsw-hart-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxint8matmul_accumulate-ctest-build)
