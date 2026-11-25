set(MLIR_TEST_PREFIX ball)
add_mlir_test(gemv_rounding TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-gemv-ball-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/gemv_rounding.mlir
    -lower-acc -o ${CMAKE_CURRENT_BINARY_DIR}/instructions.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/gemv_rounding.mlir
    --check-prefix=LOWER --implicit-check-not=mxint8gemv_ball
    -DFUNCT=${DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_BALLDOMAIN_BALLS_mxint8gemv_INSTRUCTIONS_MXINT8GEMV_FUNCT7}
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/instructions.mlir
  DEPENDS gemv_rounding.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-ball_gemv_rounding-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-gemv-ball-filecheck)
