# Host programs of the MxmmBall and MxquantBall MLIR tests: each embeds the
# thread image thread/mlir built for the test, runs it on the attention thread
# of every core and checks what it wrote.
set(POLY_MLIR_HOST_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_MLIR_HOST_DEFINES PREPEND "-D")
set(POLY_MLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../thread/mlir")
enable_language(ASM)

# ball_mxmm_mxfp8_chain
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_chain_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_mxfp8_chain_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_chain_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-multicore-baremetal
    DEPENDS
        ball_mxmm_mxfp8_chain_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_chain_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-ctest-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_chain-ctest-build)

# ball_mxmm_mxfp8_window
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_window_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_mxfp8_window_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_window_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-multicore-baremetal
    DEPENDS
        ball_mxmm_mxfp8_window_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_window_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-ctest-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_mxfp8_window-ctest-build)

# ball_mxmm_fma32
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_fma32_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_fma32_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_fma32_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-multicore-baremetal
    DEPENDS
        ball_mxmm_fma32_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_fma32_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-ctest-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_fma32-ctest-build)

# ball_mxmm_f32
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_f32_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_f32_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_f32_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-multicore-baremetal
    DEPENDS
        ball_mxmm_f32_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_f32_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-ctest-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxmm_f32-ctest-build)

# ball_mxquant_two_blocks
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ball_mxquant_two_blocks_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxquant_two_blocks_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxquant_two_blocks_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-multicore-baremetal
    DEPENDS
        ball_mxquant_two_blocks_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxquant_two_blocks_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-ctest-build ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_ball_mxquant_two_blocks-ctest-build)

# bank_mxmm_mxfp8_chain
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_chain_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_mxfp8_chain_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_chain_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-multicore-baremetal
    DEPENDS
        bank_mxmm_mxfp8_chain_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_chain_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_chain-ctest-build)

# bank_mxmm_mxfp8_window
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_window_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_mxfp8_window_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_window_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-multicore-baremetal
    DEPENDS
        bank_mxmm_mxfp8_window_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_window_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_mxfp8_window-ctest-build)

# bank_mxmm_fma32
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_fma32_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_fma32_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_fma32_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-multicore-baremetal
    DEPENDS
        bank_mxmm_fma32_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_fma32_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_fma32-ctest-build)

# bank_mxmm_f32
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_f32_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_f32_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_f32_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-multicore-baremetal
    DEPENDS
        bank_mxmm_f32_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_f32_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxmm_f32-ctest-build)

# bank_mxquant_two_blocks
file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bank_mxquant_two_blocks_image.S
    CONTENT
        ".section .rodata.thread,\"a\",@progbits\n.balign 8\n.global thread_image, thread_image_end\nthread_image:\n.incbin \"${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin\"\nthread_image_end:\n.section .note.GNU-stack,\"\",@progbits\n"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-multicore-baremetal
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64gc -mcmodel=medany -msmall-data-limit=0
        -fno-builtin -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${BBSIM_LD} -I${POLY_MLIR_SOURCE_DIR} ${POLY_MLIR_HOST_DEFINES}
        ${DESIGN_DEFINES} -I${RUNTIME_INCLUDE_DIR} -I${WORKLOAD_LIB_DIR}
        -I${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/scheduler
        -I${BUCKYBALL_REPO_ROOT}/stack/runtime/services
        -I${BUCKYBALL_REPO_ROOT}/stack/engine/include -I${BAREMETAL_DIR}
        ${POLY_BOOT_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxquant_two_blocks_main.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxquant_two_blocks_image.S
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c -lgcc -o
        ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-multicore-baremetal
    DEPENDS
        bank_mxquant_two_blocks_main.cpp
        ${POLY_MLIR_SOURCE_DIR}/mlir.h
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxquant_two_blocks_image.S
        ${POLY_MLIR_IMAGE_DIR}/${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin
        ${POLY_BOOT_SOURCE}
        ${BBSIM_LD}
        ${BUCKYBALL_REPO_ROOT}/stack/engine/src/thread.c
        ${BUCKYBALL_REPO_ROOT}/stack/engine/include/thread.h
        ${BUCKYBALL_REPO_ROOT}/stack/runtime/services/corelink.c
        ${RUNTIME_INCLUDE_DIR}/bb_cores.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)
add_custom_target(
    ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-ctest-build
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-multicore-baremetal
)
add_dependencies(${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-ctest-build ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks-thread-build)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-main-ctest-design-mlir_bank_mxquant_two_blocks-ctest-build)
