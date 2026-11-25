set(MLIR_TEST_PREFIX ball)
add_mlir_test(intsoftmax_tail TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-intsoftmax-invalid
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/intsoftmax_invalid_width.mlir -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/intsoftmax_invalid_alias.mlir -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/intsoftmax_invalid_span.mlir -lower-acc -verify-diagnostics -o /dev/null
  DEPENDS intsoftmax_invalid_width.mlir intsoftmax_invalid_alias.mlir intsoftmax_invalid_span.mlir
    ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-ball_intsoftmax_tail-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-intsoftmax-invalid)
