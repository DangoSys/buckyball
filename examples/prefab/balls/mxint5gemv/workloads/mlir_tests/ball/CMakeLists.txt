set(MLIR_TEST_PREFIX ball)
add_mlir_test(int5_gemv_physical TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-ball-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_physical.mlir
    -lower-acc -o ${CMAKE_CURRENT_BINARY_DIR}/instructions.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_physical.mlir
    --check-prefix=LOWER --implicit-check-not=mxint5gemv_ball
    -DFUNCT=${DESIGN_PROFILES_${MLIR_TEST_TARGET}_THREAD_BALLDOMAIN_BALLS_mxint5gemv_INSTRUCTIONS_MXINT5GEMV_FUNCT7}
    --input-file=${CMAKE_CURRENT_BINARY_DIR}/instructions.mlir
  DEPENDS int5_gemv_physical.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-ball_int5_gemv_physical-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-ball-filecheck)
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-invalid-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_invalid_k.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_invalid_alias.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_invalid_cbase.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_invalid_cbase_capacity.mlir
    -lower-acc -verify-diagnostics -o /dev/null
  DEPENDS int5_invalid_k.mlir int5_invalid_alias.mlir int5_invalid_cbase.mlir int5_invalid_cbase_capacity.mlir
    ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-ball_int5_gemv_physical-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-invalid-filecheck)
