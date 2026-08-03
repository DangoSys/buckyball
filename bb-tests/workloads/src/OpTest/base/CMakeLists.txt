include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BuckyballMlirTests.cmake)

add_custom_target(base-OpTest-build)

set(BUCKYBALL_MLIR_GROUP_TARGET base-OpTest-build)
set(BUCKYBALL_MLIR_TEST_PREFIX base)

add_buckyball_mlir_contract(base_intrinsics_contract
  PASSES ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_contract(base_bank_contract
  PASSES
    "--assign-physical-banks=bank_num=${BUCKYBALL_MLIR_BANK_NUM}"
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS})

add_buckyball_mlir_optest(base_mvin_mvout_16x16_i8
  PASSES
    ${BUCKYBALL_LOWER_BUCKYBALL}
    -convert-linalg-to-loops
    -expand-strided-metadata
    -lower-affine
    -convert-scf-to-cf
    -convert-cf-to-llvm
    -convert-arith-to-llvm
    -convert-math-to-llvm
    -finalize-memref-to-llvm
    -convert-func-to-llvm
    -reconcile-unrealized-casts)

add_buckyball_mlir_optest(base_bank_mvin_mvout_16x16_i8)
