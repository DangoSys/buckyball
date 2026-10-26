# ===----------------------------------------------------------------------===
# The reduce core: a CPU core with a reduction accelerator; jobs run on the core.
#
#   bbdev stack --build "--design <design>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb) and CORE. Products:
#   execute.o                runs jobs on the core itself; links into the device runtime
#   lib<target>-compiler.so  buddy-opt plugin: Buckyball dialect + the Balls of
#                            this core's accelerator
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(reduce-core LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(TARGET ${DESIGN_CORE_TARGET_${CORE}})
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# ===----------------------------------------------------------------------===
# Execution: jobs run on the core itself.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT execute.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_CORE_SIGNATURE_${CORE}}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o execute.o
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS execute.o)

# ===----------------------------------------------------------------------===
# Compiler: the Buckyball dialect and passes with the Balls of the accelerator.
# ===----------------------------------------------------------------------===
set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${TARGET}})
set(BUCKYBALL_TARGET_TABLE "${DESIGN_ISA_DIR}/${TARGET}/target.inc")

add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${TARGET}-compiler MODULE $<TARGET_OBJECTS:buckyball-compiler>)
set_target_properties(${TARGET}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
