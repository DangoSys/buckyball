# Host compiler plugin for this design's core.
set(ENGINE "${DESIGN_REPO}/stack/engine")
set(TARGET ${DESIGN_CORE_TARGET_${CORE}})

set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${TARGET}})
set(BUCKYBALL_TARGET_TABLE "${DESIGN_ISA_DIR}/${TARGET}/target.inc")
# The core owns computation recognition and its conversion to bank dataflow.
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${TARGET}-compiler MODULE $<TARGET_OBJECTS:buckyball-compiler>)
set_target_properties(${TARGET}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(CHECK "${DESIGN_MLIR}/../../../bin/FileCheck")
set(OPT "${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=$<TARGET_FILE:${TARGET}-compiler> --load-pass-plugin=$<TARGET_FILE:${TARGET}-compiler>")
set(TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
add_custom_target(${TARGET}-compiler-check
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-custom-ops.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-custom-ops.mlir --implicit-check-not=buckyball."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-int32-bias.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-int32-bias.mlir --implicit-check-not=buckyball."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-linear-chain.mlir -form-custom-ops | ${CHECK} ${TESTS}/form-linear-chain.mlir --check-prefix=FORM --implicit-check-not=custom.chain"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-linear-chain.mlir -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${TESTS}/form-linear-chain.mlir --check-prefix=BANK"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-frontend-types.mlir -allow-unregistered-dialect -lower-frontend -split-input-file -verify-diagnostics | ${CHECK} ${TESTS}/lower-frontend-types.mlir --check-prefix=VALID --implicit-check-not=frontend."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-custom-ops.mlir -convert-linalg-to-buckyball | ${CHECK} ${TESTS}/lower-custom-ops.mlir --implicit-check-not=custom.op"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/unknown-custom-op.mlir -convert-linalg-to-buckyball -verify-diagnostics -o /dev/null"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-quantized-chain.mlir -form-custom-ops | ${CHECK} ${TESTS}/form-quantized-chain.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-quantized-chain.mlir -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${TESTS}/form-quantized-chain.mlir --check-prefix=BANK --implicit-check-not=buckyball.bank_mvout"
  COMMAND bash -o pipefail -c "${OPT} ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/device/core/acc/mlir_tests/core/mega_conv2d.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/device/core/acc/mlir_tests/core/mega_conv2d.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/device/core/acc/mlir_tests/core/mega_conv2d.mlir -allow-unregistered-dialect -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/device/core/acc/mlir_tests/core/mega_conv2d.mlir --check-prefix=BANK --implicit-check-not=buckyball.bank_mvout"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-depthwise.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-depthwise.mlir --implicit-check-not=linalg.depthwise"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-bank-updates.mlir -lower-buckyball | ${CHECK} ${TESTS}/lower-bank-updates.mlir"
  DEPENDS ${TARGET}-compiler
  VERBATIM)
