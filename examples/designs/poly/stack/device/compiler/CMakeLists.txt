# ===----------------------------------------------------------------------===
# The compiler of the poly device: the framework midend (the dialects,
# lower-device, which lowers the launches and transfers to the job runtime and
# the poly device runtime, and lower-frontend-fake-op) and the poly pass that
# places a program on the cores, as the buddy-opt plugin
# libdevice-compiler.so. It compiles the device program and the graphs of the
# reduce core, which has no hardware thread; the graphs of the poly cores go
# through the core compiler (core/compiler).
#
# Built by the stack top of the device simulation for the host, configured
# with DESIGN_CMAKE. Passes (src/):
#   convert-linalg-to-device  launches on cores sharing coherent DDR
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device-compiler LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(BUCKYBALL_BALLS)
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(device-compiler MODULE)
target_link_libraries(device-compiler PRIVATE buckyball-compiler)
set_target_properties(device-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# The placement regression needs two compute ranks.
if(DESIGN_DEVICE_CORES_2_COMPONENT STREQUAL "poly")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.checked
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/device/core/compiler/coherent_placement.mlir
            -convert-linalg-to-device=design=${DESIGN_PB}
            -o ${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/device/core/compiler/coherent_placement.mlir
            --check-prefix=PLACE --implicit-check-not=device.transfer
            --input-file=${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.mlir
    COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
            --load-dialect-plugin=$<TARGET_FILE:device-compiler>
            --load-pass-plugin=$<TARGET_FILE:device-compiler>
            ${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.mlir
            "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
            "-buffer-results-to-out-params=modify-public-functions hoist-static-allocs"
            -lower-device -o ${CMAKE_CURRENT_BINARY_DIR}/coherent-lowered.mlir
    COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/device/core/compiler/coherent_placement.mlir
            --check-prefix=LOWER --input-file=${CMAKE_CURRENT_BINARY_DIR}/coherent-lowered.mlir
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.checked
    DEPENDS device-compiler ${DESIGN_PB}
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../workloads/device/core/compiler/coherent_placement.mlir
    VERBATIM)
  add_custom_target(coherent-placement-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/coherent-placement.checked)
endif()
