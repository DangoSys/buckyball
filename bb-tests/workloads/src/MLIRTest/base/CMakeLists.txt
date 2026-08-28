add_custom_target(base-mlirtest-build)

set(BUCKYBALL_MLIR_GROUP_TARGET base-mlirtest-build)
set(BUCKYBALL_MLIR_TEST_PREFIX base)

foreach(BUCKYBALL_TARGET IN LISTS BUCKYBALL_BANK_TARGETS)
  add_buckyball_mlir_contract(base_intrinsics_contract
    TARGET ${BUCKYBALL_TARGET}
    PASSES ${BUCKYBALL_LOWER_BUCKYBALL})

  add_buckyball_mlir_contract(base_bank_contract
    TARGET ${BUCKYBALL_TARGET}
    PASSES
      -assign-physical-banks
      ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS})

  add_buckyball_mlir_test(base_mvin_mvout_16x16_i8
    TARGET ${BUCKYBALL_TARGET}
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

  add_buckyball_mlir_test(base_bank_mvin_mvout_16x16_i8
    TARGET ${BUCKYBALL_TARGET})
endforeach()
