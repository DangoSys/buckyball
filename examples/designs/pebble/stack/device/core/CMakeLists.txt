# ===----------------------------------------------------------------------===
# Pebble's own core: a CPU with its Ball accelerator.
#
# Built by the stack top of each simulation entry, configured with DESIGN_CMAKE
# and CORE. Products:
#   execute.o              how the core runs jobs (direct execution)
#   libpebble-compiler.so  buddy-opt plugin: the core's fusion
#                          (form-custom-ops) and lower-buckyball around the
#                          framework bank layer
#   pebble-compiler-check  FileCheck tests of that plugin
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(pebble-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(TARGET ${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_${CORE}}_PROFILE})
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the core runs jobs.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_${CORE}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# The core's compiler plugin. The core owns computation recognition and its
# conversion to bank dataflow.
set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${TARGET}})
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${TARGET}-compiler MODULE)
target_link_libraries(${TARGET}-compiler PRIVATE buckyball-compiler)
set_target_properties(${TARGET}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(CHECK "${DESIGN_MLIR}/../../../bin/FileCheck")
set(OPT "${DESIGN_COMPILER}/bin/buddy-opt --load-dialect-plugin=$<TARGET_FILE:${TARGET}-compiler> --load-pass-plugin=$<TARGET_FILE:${TARGET}-compiler> --buckyball-design-pb=${DESIGN_PB} --buckyball-target=${TARGET}")
set(TESTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(CORE_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/device/core/acc/mlir_tests/core")
add_custom_target(${TARGET}-compiler-check
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-custom-ops.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-custom-ops.mlir --implicit-check-not=buckyball."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-int32-bias.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-int32-bias.mlir --implicit-check-not=buckyball."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-frontend-types.mlir -allow-unregistered-dialect -lower-frontend -split-input-file -verify-diagnostics | ${CHECK} ${TESTS}/lower-frontend-types.mlir --check-prefix=VALID --implicit-check-not=frontend."
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-custom-ops.mlir -convert-linalg-to-buckyball | ${CHECK} ${TESTS}/lower-custom-ops.mlir --implicit-check-not=custom.op"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/unknown-custom-op.mlir -convert-linalg-to-buckyball -verify-diagnostics -o /dev/null"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${CORE_TESTS}/mega_conv2d.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d.mlir -allow-unregistered-dialect -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${CORE_TESTS}/mega_conv2d.mlir --check-prefix=BANK"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/linear_chain_i8.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${CORE_TESTS}/linear_chain_i8.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/linear_chain_i8.mlir -allow-unregistered-dialect -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${CORE_TESTS}/linear_chain_i8.mlir --check-prefix=BANK"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d_resident.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${CORE_TESTS}/mega_conv2d_resident.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d_resident.mlir -allow-unregistered-dialect -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${CORE_TESTS}/mega_conv2d_resident.mlir --check-prefix=BANK"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d_flatten_linear.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${CORE_TESTS}/mega_conv2d_flatten_linear.mlir --check-prefix=FORM"
  COMMAND bash -o pipefail -c "${OPT} ${CORE_TESTS}/mega_conv2d_flatten_linear.mlir -allow-unregistered-dialect -form-custom-ops -convert-linalg-to-buckyball | ${CHECK} ${CORE_TESTS}/mega_conv2d_flatten_linear.mlir --check-prefix=BANK"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/form-depthwise.mlir -allow-unregistered-dialect -form-custom-ops | ${CHECK} ${TESTS}/form-depthwise.mlir --implicit-check-not=linalg.depthwise"
  COMMAND bash -o pipefail -c "${OPT} ${TESTS}/lower-bank-updates.mlir -lower-buckyball | ${CHECK} ${TESTS}/lower-bank-updates.mlir"
  DEPENDS ${TARGET}-compiler
  VERBATIM)
