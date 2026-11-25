set(MLIR_TEST_PREFIX ball)
add_mlir_test(mxint8unpack_single TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mxint8unpack-invalid
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}}
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8unpack_invalid_alias.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}}
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8unpack_invalid_zero.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}}
    ${CMAKE_CURRENT_SOURCE_DIR}/mxint8unpack_invalid_capacity.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  DEPENDS mxint8unpack_invalid_alias.mlir mxint8unpack_invalid_zero.mlir
    mxint8unpack_invalid_capacity.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-ball_mxint8unpack_single-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-mxint8unpack-invalid)
