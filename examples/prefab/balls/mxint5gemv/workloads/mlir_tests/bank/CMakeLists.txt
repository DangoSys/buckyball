set(MLIR_TEST_PREFIX bank)
add_mlir_test(int5_gemv_state TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-bank-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_state.mlir
    -assign-physical-banks -o ${CMAKE_CURRENT_BINARY_DIR}/physical.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_state.mlir
    --check-prefix=BANK --input-file=${CMAKE_CURRENT_BINARY_DIR}/physical.mlir
  DEPENDS int5_gemv_state.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-bank_int5_gemv_state-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-bank-filecheck)

add_mlir_test(int5_gemv_slots TARGET ${MLIR_TEST_TARGET})
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-slots-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_slots.mlir
    -assign-physical-banks -canonicalize -o ${CMAKE_CURRENT_BINARY_DIR}/slots-physical.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/int5_gemv_slots.mlir
    --check-prefix=BANK --input-file=${CMAKE_CURRENT_BINARY_DIR}/slots-physical.mlir
  DEPENDS int5_gemv_slots.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-bank_int5_gemv_slots-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-gemv-slots-filecheck)
add_custom_target(${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-nested-handles-filecheck
  COMMAND ${DESIGN_TARGET_OPT_${MLIR_TEST_TARGET}} ${CMAKE_CURRENT_SOURCE_DIR}/int5_nested_handles.mlir
    -assign-physical-banks -canonicalize -o ${CMAKE_CURRENT_BINARY_DIR}/nested-physical.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck ${CMAKE_CURRENT_SOURCE_DIR}/int5_nested_handles.mlir
    --check-prefix=BANK --input-file=${CMAKE_CURRENT_BINARY_DIR}/nested-physical.mlir
  DEPENDS int5_nested_handles.mlir ${DESIGN_TARGET_PLUGIN_${MLIR_TEST_TARGET}} ${DESIGN_PB}
  VERBATIM)
add_dependencies(${DESIGN_NAME}-${MLIR_TEST_TARGET}-mlirtest-bank_int5_gemv_slots-baremetal-build
  ${DESIGN_NAME}-${MLIR_TEST_TARGET}-int5-nested-handles-filecheck)
