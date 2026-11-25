# ===----------------------------------------------------------------------===
# Goban's own hart: a CPU that runs the matmul hardware thread.
#
# Built by the stack top of each simulation entry after the hart's threads,
# configured with DESIGN_CMAKE and HART. Products:
#   execute.o            how the hart runs jobs (through the kernel)
#   libhart-compiler.so  buddy-opt plugin of the hart level: the framework
#                        midend (the dialects, lower-frontend-fake-op,
#                        lower-hart: launches to kernel jobs) and the goban
#                        passes, each in its folder
#                          convert-linalg-to-hart  FP32 row regions as
#                                                  operators of the GP domain
#                                                  (bbacc.custom row_operator);
#                                                  INT8 quantization,
#                                                  decoding, products and row
#                                                  operators on the matmul
#                                                  threads, as subgraphs
#                                                  launched as early as their
#                                                  inputs allow, in turn on
#                                                  the threads; transfers to
#                                                  and from them
#                          convert-linalg-to-acc   their Balls, GP domain
#                                                  launches and the entry of
#                                                  the operator image
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(goban-hart-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(KERNEL "${DESIGN_REPO}/stack/kernel")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the hart runs jobs; instruction streams use the framework kernel.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    "-I${KERNEL}/include" -c "${RUNTIME}/execute/kernel.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${RUNTIME}/execute/kernel.c" "${RUNTIME}/scheduler/runtime.h"
    "${KERNEL}/include/kernel.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# The compiler of the graphs on this hart, including the matmul thread's
# Ball dialects and the design's acc lowering.
file(GLOB BUCKYBALL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
find_library(MATH_TRANSFORMS MLIRMathTransforms
  PATHS "${DESIGN_MLIR}/../.." NO_DEFAULT_PATH REQUIRED)
add_library(hart-compiler MODULE)
target_link_libraries(hart-compiler PRIVATE buckyball-compiler "${MATH_TRANSFORMS}")
set_target_properties(hart-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
