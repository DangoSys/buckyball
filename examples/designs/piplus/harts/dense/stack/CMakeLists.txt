# ===----------------------------------------------------------------------===
# The piplus dense hart: a CPU that runs its dense hardware thread.
#
# Built by the stack top of each simulation entry, configured with
# DESIGN_CMAKE and HART. Products:
#   libhart-compiler.so  buddy-opt plugin for the host: the framework midend (the
#                      dialects, lower-hart, lower-frontend-fake-op) and the
#                      pass of this hart (convert-linalg-to-hart/):
#     convert-linalg-to-hart  softmax over rows as SoftmaxBall work, the other
#                             FP32 row regions as operators of the GP domain;
#                             MXINT8 quantization, matrix products, softmax and
#                             row operators on the thread, as subgraphs
#                             launched as early as their inputs allow;
#                             transfers to and from it
#   execute.o          how the hart runs jobs (through the kernel)
#   operators.ld         (source) the layout of the GP operator image of a graph
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-dense-hart LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(KERNEL "${DESIGN_REPO}/stack/kernel")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# ===----------------------------------------------------------------------===
# Compiler
# ===----------------------------------------------------------------------===
set(BUCKYBALL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/ConvertLinalgToHart.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineOperators.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineReductions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineSoftmax.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineAttention.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/QuantizeBeforeViews.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/FuseBias.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/SplitRows.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineLayerNorm.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
# The polynomial approximations of convert-linalg-to-hart, which buddy-opt does not link.
find_library(MATH_TRANSFORMS MLIRMathTransforms
  PATHS "${DESIGN_MLIR}/../.." NO_DEFAULT_PATH REQUIRED)
add_library(hart-compiler MODULE)
target_link_libraries(hart-compiler PRIVATE buckyball-compiler "${MATH_TRANSFORMS}")
set_target_properties(hart-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ===----------------------------------------------------------------------===
# Execution: how the hart runs jobs; instruction streams use the framework kernel.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    "-I${KERNEL}/include" -c "${RUNTIME}/execute/kernel.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${RUNTIME}/execute/kernel.c" "${RUNTIME}/scheduler/runtime.h"
    "${KERNEL}/include/kernel.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")
