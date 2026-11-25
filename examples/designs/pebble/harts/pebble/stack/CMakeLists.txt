# ===----------------------------------------------------------------------===
# Pebble's own hart: a CPU with its Ball accelerator.
#
# Built by the stack top of each simulation entry, configured with DESIGN_CMAKE
# and HART. Products:
#   execute.o              how the hart runs jobs (direct execution)
#   libhart-compiler.so    buddy-opt plugin: the hart's recognition and
#                          fusion (convert-linalg-to-acc) on the framework
#                          midend (bank layer, lower-acc)
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(pebble-hart-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_PROFILE})
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the hart runs jobs.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# The hart's compiler plugin. The hart owns computation recognition and its
# conversion to bank dataflow.
file(GLOB_RECURSE BUCKYBALL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/*.cpp")
add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
add_library(hart-compiler MODULE)
target_link_libraries(hart-compiler PRIVATE buckyball-compiler)
set_target_properties(hart-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
