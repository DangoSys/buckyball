# ===----------------------------------------------------------------------===
# Poly's own core: a CPU that runs the attention and ffn hardware threads.
#
# Built by the stack top of each simulation entry after the core's threads,
# configured with DESIGN_CMAKE and CORE. Products:
#   execute.o              how the core runs jobs (execute.c, through the engine)
#   libengine-compiler.so  buddy-opt pass plugin of the framework engine
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-core-stack LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(ENGINE "${DESIGN_REPO}/stack/engine")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# How the core runs jobs; instruction streams use the framework engine.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_${CORE}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    "-I${ENGINE}/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" "${RUNTIME}/scheduler/runtime.h"
    "${ENGINE}/include/engine.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")

# The engine's compiler plugin, for the threads this core runs.
add_subdirectory("${ENGINE}/compiler" engine-compiler)
add_library(engine-compiler-plugin MODULE $<TARGET_OBJECTS:engine-compiler>)
set_target_properties(engine-compiler-plugin PROPERTIES OUTPUT_NAME engine-compiler
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
