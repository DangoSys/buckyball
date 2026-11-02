# ===----------------------------------------------------------------------===
# Goban's own core: a CPU that runs the matmul hardware thread.
#
# Built by the stack top of each simulation entry after the core's threads,
# configured with DESIGN_CMAKE and CORE. Products:
#   execute.o            how the core runs jobs (through the engine)
#   libcore-compiler.so  buddy-opt plugin of the core level: the framework
#                        midend (the dialects, lower-frontend-fake-op,
#                        lower-core: launches to engine jobs) and the goban
#                        pass in compiler/
#                          convert-linalg-to-core  INT8 quantization,
#                                                  decoding and products on
#                                                  the matmul threads, as
#                                                  subgraphs launched as early
#                                                  as their inputs allow, in
#                                                  turn on the threads;
#                                                  transfers to and from them
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(goban-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(ENGINE "${DESIGN_REPO}/stack/engine")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the core runs jobs; instruction streams use the framework engine.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_${CORE}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    "-I${ENGINE}/include" -c "${RUNTIME}/execute/engine.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${RUNTIME}/execute/engine.c" "${RUNTIME}/scheduler/runtime.h"
    "${ENGINE}/include/engine.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# The compiler of the graphs on this core, including the matmul thread's
# Ball dialects and the design's acc lowering.
set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_matmul})
file(GLOB BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compiler/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" midend)
add_library(core-compiler MODULE)
target_link_libraries(core-compiler PRIVATE buckyball-compiler)
set_target_properties(core-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
