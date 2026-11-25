# The MLIR tests of the Balls of poly's threads (Mxint8quantizeBall,
# Mxint8matmulBall), as thread programs: each test MLIR
# is compiled for the attention thread and linked with its entry into a
# thread image (start.S, code.ld of the kernel).
# kernel/mlir embeds every image into a host program that runs it on the
# thread of each hart and checks the result.
set(POLY_MLIR_THREAD_DEFINES ${DESIGN_TARGET_DEFS_attention})
list(TRANSFORM POLY_MLIR_THREAD_DEFINES PREPEND "-D")
string(JOIN " " POLY_MLIR_OPT ${DESIGN_TARGET_OPT_attention})
set(POLY_MLIR_BIN "${DESIGN_COMPILER}/bin")
set(POLY_MLIR_KERNEL "${BUCKYBALL_REPO_ROOT}/stack/kernel/program")
set(MXINT8QUANTIZE_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8quantize_SOURCE_PATH}/workloads/mlir_tests")
set(MXINT8MATMUL_TESTS "${BUCKYBALL_REPO_ROOT}/${DESIGN_PROFILES_attention_THREAD_BALLDOMAIN_BALLS_mxint8matmul_SOURCE_PATH}/workloads/mlir_tests")

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
        -Wl,-T,${POLY_MLIR_KERNEL}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_KERNEL}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8quantize_random_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8quantize_random.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxint8quantize_random.bin
    DEPENDS
        bank_mxint8quantize_random_entry.c
        mlir.h
        bank_mxint8quantize_random.o
        ${POLY_MLIR_KERNEL}/start.S
        ${POLY_MLIR_KERNEL}/code.ld
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
        -Wl,-T,${POLY_MLIR_KERNEL}/code.ld "$<LIST:TRANSFORM,$<TARGET_PROPERTY:bbsw-thread-attention-baremetal,INTERFACE_INCLUDE_DIRECTORIES>,PREPEND,-I>"
        ${POLY_MLIR_THREAD_DEFINES} ${POLY_MLIR_KERNEL}/start.S
        ${CMAKE_CURRENT_SOURCE_DIR}/bank_mxint8matmul_accumulate_entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/bank_mxint8matmul_accumulate.o -o ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.elf
    COMMAND
        ${ELF_OBJCOPY} -O binary --only-section=.text ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.elf
        ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin
    DEPENDS
        bank_mxint8matmul_accumulate_entry.c
        mlir.h
        bank_mxint8matmul_accumulate.o
        ${POLY_MLIR_KERNEL}/start.S
        ${POLY_MLIR_KERNEL}/code.ld
    COMMAND_EXPAND_LISTS
    VERBATIM
)
add_custom_target(${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate-thread-build DEPENDS ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate.bin)
add_dependencies(poly-ctests-build ${DESIGN_NAME}-attention-mlirtest-bank_mxint8matmul_accumulate-thread-build)
set(POLY_MLIR_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
