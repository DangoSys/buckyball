set(MLIR_TEST_PREFIX bank)
add_mlir_test(gemv_state TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-gemv-bank-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/gemv_state.mlir
    -assign-physical-banks -o ${CMAKE_CURRENT_BINARY_DIR}/physical.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gemv_state.mlir
    --check-prefix=BANK --input-file=${CMAKE_CURRENT_BINARY_DIR}/physical.mlir
  DEPENDS gemv_state.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-bank_gemv_state-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-gemv-bank-filecheck)
