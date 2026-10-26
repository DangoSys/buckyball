add_custom_target(base-mlirtest-build)

set(BUCKYBALL_MLIR_GROUP_TARGET base-mlirtest-build)
set(BUCKYBALL_MLIR_TEST_PREFIX base)

foreach(BUCKYBALL_TARGET IN LISTS DESIGN_CPU_TARGETS)
  if(NOT BUCKYBALL_TARGET IN_LIST DESIGN_BANK_TARGETS)
    continue()
  endif()
  add_buckyball_mlir_test(base_memref_copy_dynamic TARGET ${BUCKYBALL_TARGET})
  add_buckyball_mlir_test(base_mvin_mvout_16x16_i8 TARGET ${BUCKYBALL_TARGET})
  add_buckyball_mlir_test(base_bank_mvin_mvout_16x16_i8 TARGET ${BUCKYBALL_TARGET})
endforeach()
