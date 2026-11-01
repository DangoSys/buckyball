# The C tests of the MXINT8 Balls of poly's threads, as thread programs: each
# test's program (<ball>/workloads/ctests/<test>.c) is linked into a thread
# image (start.S, code.ld of the engine). engine/balls embeds every image into
# a host program that runs it on the thread of each core and checks the
# result against the vectors.h of its Ball. The quantize tests run on the
# attention thread, the matmul tests on the ffn thread.
set(POLY_BALL_ENGINE "${BUCKYBALL_REPO_ROOT}/stack/engine/program")
set(POLY_BALL_ATTENTION_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_BALL_ATTENTION_DEFINES PREPEND "-D")
set(POLY_BALL_FFN_DEFINES ${DESIGN_TARGET_DEFS_ffn})
list(TRANSFORM POLY_BALL_FFN_DEFINES PREPEND "-D")
set(MXINT8QUANTIZE_WORKLOADS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads")
set(MXINT8MATMUL_WORKLOADS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_ffn_THREAD_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads")

# mxint8quantize_quant_edges
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.elf ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8QUANTIZE_WORKLOADS} ${POLY_BALL_ATTENTION_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/quant_edges.c -o ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.elf ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.bin
    DEPENDS
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/quant_edges.c
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/task.h
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/vectors.h
        ${MXINT8QUANTIZE_WORKLOADS}/isa/mxint8quantize.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges-thread-build DEPENDS ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_edges-thread-build)

# mxint8quantize_quant_random
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.elf ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8QUANTIZE_WORKLOADS} ${POLY_BALL_ATTENTION_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/quant_random.c -o ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.elf ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.bin
    DEPENDS
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/quant_random.c
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/task.h
        ${MXINT8QUANTIZE_WORKLOADS}/ctests/vectors.h
        ${MXINT8QUANTIZE_WORKLOADS}/isa/mxint8quantize.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random-thread-build DEPENDS ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-ctest-mxint8quantize_quant_random-thread-build)

# mxint8matmul_mm_small
add_custom_command(
    OUTPUT ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8MATMUL_WORKLOADS} ${POLY_BALL_FFN_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_small.c -o ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.bin
    DEPENDS
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_small.c
        ${MXINT8MATMUL_WORKLOADS}/ctests/task.h
        ${MXINT8MATMUL_WORKLOADS}/ctests/vectors.h
        ${MXINT8MATMUL_WORKLOADS}/isa/mxint8matmul.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small-thread-build DEPENDS ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_small-thread-build)

# mxint8matmul_mm_accumulate
add_custom_command(
    OUTPUT ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8MATMUL_WORKLOADS} ${POLY_BALL_FFN_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_accumulate.c -o ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.bin
    DEPENDS
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_accumulate.c
        ${MXINT8MATMUL_WORKLOADS}/ctests/task.h
        ${MXINT8MATMUL_WORKLOADS}/ctests/vectors.h
        ${MXINT8MATMUL_WORKLOADS}/isa/mxint8matmul.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate-thread-build DEPENDS ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_accumulate-thread-build)

# mxint8matmul_mm_extremes
add_custom_command(
    OUTPUT ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8MATMUL_WORKLOADS} ${POLY_BALL_FFN_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_extremes.c -o ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.bin
    DEPENDS
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_extremes.c
        ${MXINT8MATMUL_WORKLOADS}/ctests/task.h
        ${MXINT8MATMUL_WORKLOADS}/ctests/vectors.h
        ${MXINT8MATMUL_WORKLOADS}/isa/mxint8matmul.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes-thread-build DEPENDS ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_extremes-thread-build)

# mxint8matmul_mm_long
add_custom_command(
    OUTPUT ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_BALL_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-ffn-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        -I${MXINT8MATMUL_WORKLOADS} ${POLY_BALL_FFN_DEFINES} ${POLY_BALL_ENGINE}/start.S
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_long.c -o ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.elf ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.bin
    DEPENDS
        ${MXINT8MATMUL_WORKLOADS}/ctests/mm_long.c
        ${MXINT8MATMUL_WORKLOADS}/ctests/task.h
        ${MXINT8MATMUL_WORKLOADS}/ctests/vectors.h
        ${MXINT8MATMUL_WORKLOADS}/isa/mxint8matmul.h
        ${POLY_BALL_ENGINE}/start.S
        ${POLY_BALL_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long-thread-build DEPENDS ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-ffn-ctest-mxint8matmul_mm_long-thread-build)

set(POLY_BALL_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
