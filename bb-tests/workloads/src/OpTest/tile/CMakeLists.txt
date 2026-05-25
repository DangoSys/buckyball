set(BBSIM_LD ${BBSW_BAREMETAL_DIR}/bbsim.ld)
set(C_FLAGS -g -fno-common -O2 -static -march=rv64gc -mcmodel=medany
    -fno-builtin-printf -specs=nano.specs -specs=nosys.specs -nostartfiles
    -Wl,-T,${BBSIM_LD})

# Compile matmul.mlir -> matmul.o
add_custom_command(
    OUTPUT matmul.o
    COMMAND ${BUDDY_OPT} ${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir
        -convert-tile-to-buckyball
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
    COMMENT "Building matmul.o from Tile MLIR"
    VERBATIM
)

# Link baremetal binary
add_custom_command(
    OUTPUT optest_tile_matmul_singlecore-baremetal
    COMMAND ${ELF_CC} ${C_FLAGS}
        -o ${CMAKE_CURRENT_BINARY_DIR}/optest_tile_matmul_singlecore-baremetal
        ${BBSW_BAREMETAL_DIR}/crt0.S
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.o
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_main.c
        ${BBSW_BAREMETAL_DIR}/crt0.S
    COMMENT "Linking optest_tile_matmul_singlecore-baremetal"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

add_custom_target(optest_tile_matmul_singlecore_baremetal ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optest_tile_matmul_singlecore-baremetal
)
