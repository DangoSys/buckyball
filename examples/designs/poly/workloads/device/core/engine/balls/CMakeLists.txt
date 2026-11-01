# Host programs of the MXINT8 Ball C tests: each embeds the thread image
# thread/balls built for the test, runs it on the attention (quantize) or ffn
# (matmul) thread of every core and checks what it wrote against the
# vectors.h of its Ball.
set(POLY_BALL_ATTENTION_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_BALL_ATTENTION_DEFINES PREPEND "-D")
set(POLY_BALL_FFN_DEFINES ${DESIGN_TARGET_DEFS_ffn})
list(TRANSFORM POLY_BALL_FFN_DEFINES PREPEND "-D")
set(MXINT8QUANTIZE_CTESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads/ctests")
set(MXINT8MATMUL_CTESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_ffn_THREAD_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads/ctests")

# mxint8quantize_quant_edges
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_edges_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8QUANTIZE_CTESTS} ${POLY_BALL_ATTENTION_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8quantize_quant_edges_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_edges_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal
    DEPENDS
        mxint8quantize_quant_edges_main.c
        ${MXINT8QUANTIZE_CTESTS}/task.h
        ${MXINT8QUANTIZE_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_edges_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal-build ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_edges-multicore-baremetal-build)

# mxint8quantize_quant_random
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_random_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8QUANTIZE_CTESTS} ${POLY_BALL_ATTENTION_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8quantize_quant_random_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_random_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal
    DEPENDS
        mxint8quantize_quant_random_main.c
        ${MXINT8QUANTIZE_CTESTS}/task.h
        ${MXINT8QUANTIZE_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8quantize_quant_random_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal-build ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8quantize_quant_random-multicore-baremetal-build)

# mxint8matmul_mm_small
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_small_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8MATMUL_CTESTS} ${POLY_BALL_FFN_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_mm_small_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_small_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal
    DEPENDS
        mxint8matmul_mm_small_main.c
        ${MXINT8MATMUL_CTESTS}/task.h
        ${MXINT8MATMUL_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_small_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_small-multicore-baremetal-build)

# mxint8matmul_mm_accumulate
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_accumulate_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8MATMUL_CTESTS} ${POLY_BALL_FFN_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_mm_accumulate_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_accumulate_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal
    DEPENDS
        mxint8matmul_mm_accumulate_main.c
        ${MXINT8MATMUL_CTESTS}/task.h
        ${MXINT8MATMUL_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_accumulate_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_accumulate-multicore-baremetal-build)

# mxint8matmul_mm_extremes
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_extremes_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8MATMUL_CTESTS} ${POLY_BALL_FFN_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_mm_extremes_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_extremes_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal
    DEPENDS
        mxint8matmul_mm_extremes_main.c
        ${MXINT8MATMUL_CTESTS}/task.h
        ${MXINT8MATMUL_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_extremes_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_extremes-multicore-baremetal-build)

# mxint8matmul_mm_long
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_long_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        $<TARGET_PROPERTY:bbsw-start-baremetal,INTERFACE_LINK_OPTIONS> -I${MXINT8MATMUL_CTESTS} ${POLY_BALL_FFN_DEFINES}
        ${DESIGN_DEFINES} "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-device-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-core-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>" "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/mxint8matmul_mm_long_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_long_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        $<TARGET_FILE:bbsw-core-baremetal> -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal
    DEPENDS
        mxint8matmul_mm_long_main.c
        ${MXINT8MATMUL_CTESTS}/task.h
        ${MXINT8MATMUL_CTESTS}/vectors.h
        ${CMAKE_CURRENT_BINARY_DIR}/mxint8matmul_mm_long_image.S
        ${POLY_BALL_IMAGE_DIR}/${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.bin
        ${POLY_BOOT_SOURCE}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        $<TARGET_FILE:bbsw-core-baremetal>
        ${BUCKYBALL_BBSW_HEADERS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal-build DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mxint8matmul_mm_long-multicore-baremetal-build)
