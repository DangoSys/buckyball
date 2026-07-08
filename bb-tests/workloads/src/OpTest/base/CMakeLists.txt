include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BuckyballMlirTests.cmake)

add_custom_target(base-OpTest-build)

set(BUCKYBALL_MLIR_GROUP_TARGET base-OpTest-build)
set(BUCKYBALL_MLIR_TEST_PREFIX base)

add_buckyball_mlir_contract(base_intrinsics_contract
  PASSES ${BUCKYBALL_LOWER_BUCKYBALL})

add_buckyball_mlir_contract(base_bank_contract
  PASSES
    "--assign-physical-banks=bank_num=16"
    ${BUCKYBALL_LOWER_BANK_SSA_TO_INTRINSICS})
