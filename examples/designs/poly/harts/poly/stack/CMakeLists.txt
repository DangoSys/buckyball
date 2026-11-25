# ===----------------------------------------------------------------------===
# Poly's own hart: a CPU that runs the attention and ffn hardware threads.
#
# Built by the stack top of each simulation entry after the hart's threads,
# configured with DESIGN_CMAKE and HART. Products:
#   libhart-compiler.so  buddy-opt plugin for the host: the framework midend
#                        (the dialects, lower-hart, which lowers the launches
#                        to batches of asynchronous kernel jobs, and
#                        lower-frontend-fake-op) and the passes of this hart,
#                        each in its folder:
#     convert-linalg-to-hart  FP32 row regions as operators of the GP domain;
#                             the MXINT8 quantizations and matrix products of a
#                             graph on the attention or ffn threads, as
#                             subgraphs launched as early as their inputs
#                             allow, in turn on the threads of the kind;
#                             transfers to and from them
#     convert-linalg-to-acc   MXINT8 product and quantization mapping for the
#                             selected thread profile, including quantized outputs
#   execute.o            how the hart runs jobs (execute.c, through the kernel)
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-hart-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(KERNEL "${DESIGN_REPO}/stack/kernel")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the hart runs jobs; instruction streams use the framework kernel.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    "-I${KERNEL}/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" "${RUNTIME}/scheduler/runtime.h"
    "${KERNEL}/include/kernel.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# ===----------------------------------------------------------------------===
# Compiler
# ===----------------------------------------------------------------------===
set(BUCKYBALL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/ConvertLinalgToAcc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/ConvertLinalgToHart.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerActivation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerIntegerSoftmax.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerQuant.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerRowOperators.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineOperators.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/PartitionRowPipelines.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineReductions.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
find_library(MATH_TRANSFORMS MLIRMathTransforms
  PATHS "${DESIGN_MLIR}/../.." NO_DEFAULT_PATH REQUIRED)
add_library(hart-compiler MODULE)
target_link_libraries(hart-compiler PRIVATE buckyball-compiler "${MATH_TRANSFORMS}")
set_target_properties(hart-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:hart-compiler>
          --load-pass-plugin=$<TARGET_FILE:hart-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/materialized_outputs.mlir
          -forward-materialized-tensors -o ${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/materialized_outputs.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.mlir
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          ${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.mlir
          "-one-shot-bufferize=bufferize-function-boundaries" -o /dev/null
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.checked
  DEPENDS hart-compiler ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/materialized_outputs.mlir
  VERBATIM)
add_custom_target(materialized-outputs-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/materialized-outputs.checked)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/logical-dataflow.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:hart-compiler>
          --load-pass-plugin=$<TARGET_FILE:hart-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/logical_dataflow.mlir
          -inline -expand-strided-metadata -lower-affine -canonicalize
          "-lower-hart=design=${DESIGN_PB} hart=poly"
          -o ${CMAKE_CURRENT_BINARY_DIR}/logical-dataflow.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/logical_dataflow.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/logical-dataflow.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/logical-dataflow.checked
  DEPENDS hart-compiler ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/logical_dataflow.mlir ${DESIGN_PB}
  VERBATIM)
add_custom_target(logical-dataflow-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/logical-dataflow.checked)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/device-cleanup.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:hart-compiler>
          --load-pass-plugin=$<TARGET_FILE:hart-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/device_cleanup.mlir
          "-buffer-results-to-out-params=modify-public-functions hoist-dynamic-allocs"
          -lower-device -o ${CMAKE_CURRENT_BINARY_DIR}/device-cleanup.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/device_cleanup.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/device-cleanup.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/device-cleanup.checked
  DEPENDS hart-compiler ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/device_cleanup.mlir
  VERBATIM)
add_custom_target(device-cleanup-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/device-cleanup.checked)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shape-only-input.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:hart-compiler>
          --load-pass-plugin=$<TARGET_FILE:hart-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/shape_only_input.mlir
          "-one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
          -canonicalize -o ${CMAKE_CURRENT_BINARY_DIR}/shape-only-input.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/shape_only_input.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/shape-only-input.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/shape-only-input.checked
  DEPENDS hart-compiler ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/shape_only_input.mlir
  VERBATIM)
add_custom_target(shape-only-input-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shape-only-input.checked)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gp-host-cleanup.checked
  COMMAND ${DESIGN_COMPILER}/bin/buddy-opt
          --load-dialect-plugin=$<TARGET_FILE:hart-compiler>
          --load-pass-plugin=$<TARGET_FILE:hart-compiler>
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/gp_host_cleanup.mlir
          -prune-gp-operators -symbol-dce -o ${CMAKE_CURRENT_BINARY_DIR}/gp-host-cleanup.mlir
  COMMAND ${DESIGN_MLIR}/../../../bin/FileCheck
          ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/gp_host_cleanup.mlir
          --input-file=${CMAKE_CURRENT_BINARY_DIR}/gp-host-cleanup.mlir
  COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/gp-host-cleanup.checked
  DEPENDS hart-compiler ${CMAKE_CURRENT_SOURCE_DIR}/../workloads/compiler/gp_host_cleanup.mlir
  VERBATIM)
add_custom_target(gp-host-cleanup-check ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gp-host-cleanup.checked)
