set(BUCKYBALL_MLIR_GROUP_TARGET pebble-mlir-tests-build)
set(BUCKYBALL_MLIR_TEST_PREFIX tile)

set(PEBBLE_MEGA_REGION_PASSES
  -convert-tile-to-buckyball)

add_buckyball_mlir_contract(mega_matmul_contract
  TARGET pebble
  PASSES ${PEBBLE_MEGA_REGION_PASSES})

add_buckyball_mlir_contract(mega_matmul_bank_contract
  TARGET pebble
  PASSES
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks)

add_buckyball_mlir_contract(mega_conv2d_contract
  TARGET pebble
  PASSES ${PEBBLE_MEGA_REGION_PASSES})

add_buckyball_mlir_contract(mega_conv2d_bank_contract
  TARGET pebble
  PASSES
    -convert-tile-to-buckyball
    -lower-buckyball-to-bank-ssa
    -assign-physical-banks)
