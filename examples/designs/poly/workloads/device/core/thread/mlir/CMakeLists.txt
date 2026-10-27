# The MLIR tests of MxmmBall and MxquantBall, the Balls of poly's threads, as
# thread programs: each test MLIR is compiled for the attention thread and
# linked with its entry into a thread image (start.S, code.ld of the engine).
# engine/mlir embeds every image into a host program that runs it on the
# thread of each core and checks the result.
set(POLY_MLIR_THREAD_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_MLIR_THREAD_DEFINES PREPEND "-D")
string(JOIN " " POLY_MLIR_OPT ${DESIGN_TARGET_OPT_attention})
set(POLY_MLIR_BIN "${DESIGN_COMPILER}/bin")
set(POLY_MLIR_ENGINE "${BUCKYBALL_REPO_ROOT}/stack/engine/program")
set(MXMM_TESTS "${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxmm/workloads/mlir_tests")
set(MXQUANT_TESTS "${BUCKYBALL_REPO_ROOT}/examples/prefab/balls/mxquant/workloads/mlir_tests")

# ball_mxmm_mxfp8_chain
add_custom_command(
    OUTPUT ball_mxmm_mxfp8_chain.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_mxfp8_chain.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_mxfp8_chain.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_chain-thread-build)

# ball_mxmm_mxfp8_window
add_custom_command(
    OUTPUT ball_mxmm_mxfp8_window.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_mxfp8_window.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_mxfp8_window.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_mxfp8_window-thread-build)

# ball_mxmm_fma32
add_custom_command(
    OUTPUT ball_mxmm_fma32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_fma32.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_fma32.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_fma32-thread-build)

# ball_mxmm_f32
add_custom_command(
    OUTPUT ball_mxmm_f32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/ball/mxmm_f32.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxmm_f32.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxmm_f32-thread-build)

# ball_mxquant_two_blocks
add_custom_command(
    OUTPUT ball_mxquant_two_blocks.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXQUANT_TESTS}/ball/mxquant_two_blocks.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o ball_mxquant_two_blocks.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-ball_mxquant_two_blocks-thread-build)

# bank_mxmm_mxfp8_chain
add_custom_command(
    OUTPUT bank_mxmm_mxfp8_chain.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_mxfp8_chain.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_mxfp8_chain.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_chain-thread-build)

# bank_mxmm_mxfp8_window
add_custom_command(
    OUTPUT bank_mxmm_mxfp8_window.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_mxfp8_window.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_mxfp8_window.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_mxfp8_window-thread-build)

# bank_mxmm_fma32
add_custom_command(
    OUTPUT bank_mxmm_fma32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_fma32.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_fma32.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_fma32-thread-build)

# bank_mxmm_f32
add_custom_command(
    OUTPUT bank_mxmm_f32.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXMM_TESTS}/bank/mxmm_f32.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxmm_f32.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxmm_f32-thread-build)

# bank_mxquant_two_blocks
add_custom_command(
    OUTPUT bank_mxquant_two_blocks.o
    COMMAND
        bash -o pipefail -c
        "${POLY_MLIR_OPT} ${MXQUANT_TESTS}/bank/mxquant_two_blocks.mlir -assign-physical-banks -lower-buckyball -convert-linalg-to-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | ${POLY_MLIR_BIN}/buddy-translate --buddy-to-llvmir | ${POLY_MLIR_BIN}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -code-model=medium -mattr=+xbuckyball,+m -target-abi=lp64 --riscv-disable-using-constant-pool-for-large-ints -o bank_mxquant_two_blocks.o"
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
        -Wl,-T,${POLY_MLIR_ENGINE}/code.ld -I${WORKLOAD_LIB_DIR}
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
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxquant_two_blocks-thread-build)
set(POLY_MLIR_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
