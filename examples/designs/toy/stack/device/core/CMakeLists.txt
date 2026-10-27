# ===----------------------------------------------------------------------===
# Toy's own core: a CPU with its Ball accelerator.
#
# Built by the stack top of each simulation entry, configured with DESIGN_CMAKE
# and CORE. Products:
#   execute.o           how the core runs jobs (direct execution)
#   libtoy-compiler.so  buddy-opt plugin: the core's fusion (form-custom-ops)
#                       and lower-buckyball around the framework bank layer
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(toy-core-stack LANGUAGES C CXX)
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
