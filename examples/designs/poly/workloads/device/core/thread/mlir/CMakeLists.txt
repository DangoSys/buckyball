# The MLIR tests of the Balls of poly's threads (MxmmBall, MxquantBall,
# Mxint8quantizeBall, Mxint8matmulBall), as thread programs: each test MLIR
# is compiled for the attention thread and linked with its entry into a
# thread image (start.S, code.ld of the engine).
# engine/mlir embeds every image into a host program that runs it on the
# thread of each core and checks the result.
set(POLY_MLIR_THREAD_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_MLIR_THREAD_DEFINES PREPEND "-D")
string(JOIN " " POLY_MLIR_OPT ${DESIGN_TARGET_OPT_attention})
set(POLY_MLIR_BIN "${DESIGN_COMPILER}/bin")
set(POLY_MLIR_ENGINE "${BUCKYBALL_REPO_ROOT}/stack/engine/program")
set(MXMM_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxmm_SOURCE_PATH}/workloads/mlir_tests")
set(MXQUANT_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxquant_SOURCE_PATH}/workloads/mlir_tests")
set(MXINT8QUANTIZE_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads/mlir_tests")
set(MXINT8MATMUL_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads/mlir_tests")

# ball_mxmm_mxfp8_chain
add_custom_command(
    OUTPUT ball_mxmm_mxfp8_chain.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_mxfp8_chain.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_mxfp8_chain.o"
    DEPENDS ${MXMM_TESTS}/ball/mxmm_mxfp8_chain.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.elf ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_mxfp8_chain_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_chain.o -o ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.elf
        ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin
    DEPENDS
        ball_mxmm_mxfp8_chain_entry.c
        mlir.h
        ball_mxmm_mxfp8_chain.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain-thread-build)

# ball_mxmm_mxfp8_window
add_custom_command(
    OUTPUT ball_mxmm_mxfp8_window.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_mxfp8_window.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_mxfp8_window.o"
    DEPENDS ${MXMM_TESTS}/ball/mxmm_mxfp8_window.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.elf ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_mxfp8_window_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_mxfp8_window.o -o ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.elf
        ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin
    DEPENDS
        ball_mxmm_mxfp8_window_entry.c
        mlir.h
        ball_mxmm_mxfp8_window.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window-thread-build)

# ball_mxmm_fma32
add_custom_command(
    OUTPUT ball_mxmm_fma32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_fma32.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_fma32.o"
    DEPENDS ${MXMM_TESTS}/ball/mxmm_fma32.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.elf ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_fma32_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_fma32.o -o ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.elf
        ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin
    DEPENDS
        ball_mxmm_fma32_entry.c
        mlir.h
        ball_mxmm_fma32.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32-thread-build)

# ball_mxmm_f32
add_custom_command(
    OUTPUT ball_mxmm_f32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_f32.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_f32.o"
    DEPENDS ${MXMM_TESTS}/ball/mxmm_f32.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.elf ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxmm_f32_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxmm_f32.o -o ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.elf
        ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin
    DEPENDS
        ball_mxmm_f32_entry.c
        mlir.h
        ball_mxmm_f32.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32-thread-build)

# ball_mxquant_two_blocks
add_custom_command(
    OUTPUT ball_mxquant_two_blocks.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXQUANT_TESTS}/ball/mxquant_two_blocks.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxquant_two_blocks.o"
    DEPENDS ${MXQUANT_TESTS}/ball/mxquant_two_blocks.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.elf ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/ball_mxquant_two_blocks_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ball_mxquant_two_blocks.o -o ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.elf
        ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin
    DEPENDS
        ball_mxquant_two_blocks_entry.c
        mlir.h
        ball_mxquant_two_blocks.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks-thread-build)

# bank_mxmm_mxfp8_chain
add_custom_command(
    OUTPUT bank_mxmm_mxfp8_chain.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_mxfp8_chain.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_mxfp8_chain.o"
    DEPENDS ${MXMM_TESTS}/bank/mxmm_mxfp8_chain.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_mxfp8_chain_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_chain.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin
    DEPENDS
        bank_mxmm_mxfp8_chain_entry.c
        mlir.h
        bank_mxmm_mxfp8_chain.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain-thread-build)

# bank_mxmm_mxfp8_window
add_custom_command(
    OUTPUT bank_mxmm_mxfp8_window.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_mxfp8_window.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_mxfp8_window.o"
    DEPENDS ${MXMM_TESTS}/bank/mxmm_mxfp8_window.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_mxfp8_window_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_mxfp8_window.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin
    DEPENDS
        bank_mxmm_mxfp8_window_entry.c
        mlir.h
        bank_mxmm_mxfp8_window.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window-thread-build)

# bank_mxmm_fma32
add_custom_command(
    OUTPUT bank_mxmm_fma32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_fma32.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_fma32.o"
    DEPENDS ${MXMM_TESTS}/bank/mxmm_fma32.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_fma32_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_fma32.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin
    DEPENDS
        bank_mxmm_fma32_entry.c
        mlir.h
        bank_mxmm_fma32.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32-thread-build)

# bank_mxmm_f32
add_custom_command(
    OUTPUT bank_mxmm_f32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_f32.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_f32.o"
    DEPENDS ${MXMM_TESTS}/bank/mxmm_f32.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxmm_f32_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxmm_f32.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin
    DEPENDS
        bank_mxmm_f32_entry.c
        mlir.h
        bank_mxmm_f32.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32-thread-build)

# bank_mxquant_two_blocks
add_custom_command(
    OUTPUT bank_mxquant_two_blocks.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXQUANT_TESTS}/bank/mxquant_two_blocks.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxquant_two_blocks.o"
    DEPENDS ${MXQUANT_TESTS}/bank/mxquant_two_blocks.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxquant_two_blocks_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxquant_two_blocks.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin
    DEPENDS
        bank_mxquant_two_blocks_entry.c
        mlir.h
        bank_mxquant_two_blocks.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks-thread-build)
# bank_mxint8quantize_random
add_custom_command(
    OUTPUT bank_mxint8quantize_random.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXINT8QUANTIZE_TESTS}/bank/mxint8quantize_random.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxint8quantize_random.o"
    DEPENDS ${MXINT8QUANTIZE_TESTS}/bank/mxint8quantize_random.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8quantize_random_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8quantize_random.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin
    DEPENDS
        bank_mxint8quantize_random_entry.c
        mlir.h
        bank_mxint8quantize_random.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random-thread-build)
# bank_mxint8matmul_accumulate
add_custom_command(
    OUTPUT bank_mxint8matmul_accumulate.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXINT8MATMUL_TESTS}/bank/mxint8matmul_accumulate.mlir -assign-physical-banks -lower-acc -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxint8matmul_accumulate.o"
    DEPENDS ${MXINT8MATMUL_TESTS}/bank/mxint8matmul_accumulate.mlir ${DESIGN_TARGET_PLUGIN_attention}
    VERBATIM
)
add_custom_command(
    OUTPUT ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.elf ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin
    COMMAND
        ${ELF_CC} -O2 -g -march=rv64im -mabi=lp64 -mcmodel=medany
        -ffreestanding -fno-builtin -fno-pic -fno-stack-protector
        -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
        -nostdlib -nostartfiles -Wl,--no-relax
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_ENGINE}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8matmul_accumulate_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8matmul_accumulate.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin
    DEPENDS
        bank_mxint8matmul_accumulate_entry.c
        mlir.h
        bank_mxint8matmul_accumulate.o
        ${POLY_MLIR_ENGINE}/start.S
        ${POLY_MLIR_ENGINE}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate-thread-build)
set(POLY_MLIR_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
