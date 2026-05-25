set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD})

# Compile matmul.mlir -> matmul.o
# Dumps intermediate MLIR at each lowering stage for debugging.
add_custom_command(
    OUTPUT matmul.o
    # Stage 1: after buckyball.matmul -> bank-SSA lowering
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        > ${CMAKE_CURRENT_BINARY_DIR}/matmul_1_bank_ssa.mlir

    # Stage 2: after assign-physical-banks
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        "--assign-physical-banks=bank_num=32"
        > ${CMAKE_CURRENT_BINARY_DIR}/matmul_2_physical_banks.mlir

    # Stage 3: after lower-bank-ssa-to-intrinsics + lower-buckyball
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        "--assign-physical-banks=bank_num=32"
        -lower-bank-ssa-to-intrinsics
        -lower-buckyball
        > ${CMAKE_CURRENT_BINARY_DIR}/matmul_3_buckyball_intrinsics.mlir

    # Stage 4: after full LLVM lowering (final MLIR before translate)
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        "--assign-physical-banks=bank_num=32"
        -lower-bank-ssa-to-intrinsics
        -convert-linalg-to-loops
        -expand-strided-metadata
        -convert-scf-to-cf
        -lower-buckyball
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts
        > ${CMAKE_CURRENT_BINARY_DIR}/matmul_4_llvm.mlir

    # Stage 5: final LLVM IR
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        "--assign-physical-banks=bank_num=32"
        -lower-bank-ssa-to-intrinsics
        -convert-linalg-to-loops
        -expand-strided-metadata
        -convert-scf-to-cf
        -lower-buckyball
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir
        > ${CMAKE_CURRENT_BINARY_DIR}/matmul_5.ll

    # Final: build matmul.o
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -lower-buckyball-to-bank-ssa
        "--assign-physical-banks=bank_num=32"
        -lower-bank-ssa-to-intrinsics
        -convert-linalg-to-loops
        -expand-strided-metadata
        -convert-scf-to-cf
        -lower-buckyball
        -convert-arith-to-llvm
        -convert-math-to-llvm
        -finalize-memref-to-llvm
        -convert-func-to-llvm
        -reconcile-unrealized-casts |
    ${BUDDY_TRANSLATE} --buddy-to-llvmir |
    ${BUDDY_LLC} -filetype=obj -mtriple=riscv64 -O2 -mattr=+buddyext,+D -float-abi=hard
        -o ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
    COMMENT "Building matmul.o from Buckyball MLIR (with intermediate dumps)"
    VERBATIM
)

# Link baremetal binary
add_custom_command(
    OUTPUT optest_buckyball_matmul_singlecore-baremetal
    COMMAND ${ELF_CC} ${C_FLAGS}
        -o ${CMAKE_CURRENT_BINARY_DIR}/optest_buckyball_matmul_singlecore-baremetal
        ${BBSW_BAREMETAL_DIR}/crt0.S
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${BBSW_BAREMETAL_DIR}/crt0.S
    COMMENT "Linking optest_buckyball_matmul_singlecore-baremetal"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

add_custom_target(optest_buckyball_matmul_singlecore_baremetal ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optest_buckyball_matmul_singlecore-baremetal
)
