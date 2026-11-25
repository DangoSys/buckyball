# ===----------------------------------------------------------------------===
# The compiler of the poly device: the framework midend (the dialects,
# lower-device, which lowers the launches and transfers to the job runtime and
# the poly device runtime, and lower-frontend-fake-op) and the poly pass that
# places a program on the harts, as the buddy-opt plugin
# libdevice-compiler.so. It compiles the device program and the graphs of the
# reduce hart, which has no hardware thread; the graphs of the poly harts go
# through the hart stacks.
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes:
#   convert-linalg-to-device  launches on the harts, transfers through the device SHM
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
add_library(device-compiler MODULE)
target_link_libraries(device-compiler PRIVATE buckyball-compiler)
set_target_properties(device-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# The placement regression needs two compute ranks.
if(DESIGN_DEVICE_HARTS_2_COMPONENT STREQUAL "poly")
  set(REDUCER_COMPONENT ${DESIGN_DEVICE_HARTS_${DESIGN_WORKLOAD_CONTROL_HART}_COMPONENT})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/fanout_transfer.mlir.in
    ${CMAKE_CURRENT_BINARY_DIR}/fanout-input.mlir @ONLY)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.checked
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/transfer_placement.mlir
            -convert-linalg-to-device=design=${DESIGN_PB}
            -o ${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/transfer_placement.mlir
            --check-prefix=PLACE
            --input-file=${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.mlir
            "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -lower-device -o ${CMAKE_CURRENT_BINARY_DIR}/transfer-lowered.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/transfer_placement.mlir
            --check-prefix=LOWER --input-file=${CMAKE_CURRENT_BINARY_DIR}/transfer-lowered.mlir
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.checked
    DEPENDS device-compiler ${DESIGN_PB}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/transfer_placement.mlir
    VERBATIM)
  add_custom_target(transfer-placement-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/transfer-placement.checked)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.checked
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_BINARY_DIR}/fanout-input.mlir
            -convert-linalg-to-device=design=${DESIGN_PB}
            -o ${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_BINARY_DIR}/fanout-input.mlir
            --check-prefix=PLACE --input-file=${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.mlir
            "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -lower-device -o ${CMAKE_CURRENT_BINARY_DIR}/fanout-lowered.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_BINARY_DIR}/fanout-input.mlir
            --check-prefix=LOWER --input-file=${CMAKE_CURRENT_BINARY_DIR}/fanout-lowered.mlir
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.checked
    DEPENDS device-compiler ${DESIGN_PB}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/fanout_transfer.mlir.in
            ${CMAKE_CURRENT_BINARY_DIR}/fanout-input.mlir
    VERBATIM)
  add_custom_target(fanout-transfer-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fanout-transfer.checked)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/deferred-transfer.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:device-compiler>
          --load-pass-plugin=$<TARGET_FILE:device-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/deferred_transfer.mlir
          -lower-device -canonicalize -o ${CMAKE_CURRENT_BINARY_DIR}/deferred-transfer.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/deferred_transfer.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/deferred-transfer.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/deferred-transfer.checked
  DEPENDS device-compiler
          ${CMAKE_CURRENT_SOURCE_DIR}/../../workloads/compiler/deferred_transfer.mlir
  VERBATIM)
add_custom_target(deferred-transfer-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/deferred-transfer.checked)
