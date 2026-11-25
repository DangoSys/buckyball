# ===----------------------------------------------------------------------===
# The reduce hart: a CPU hart with a reduction accelerator; jobs run on the hart.
#
#   bbdev stack --build "--config <config> --device <device>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb) and HART. Products:
#   execute.o                runs jobs on the hart itself; links into the device runtime
#   libhart-compiler.so      buddy-opt plugin: acc dialect + the Balls of
#                            this hart's accelerator
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-reduce-unpack-hart LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(TARGET ${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_PROFILE})
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# ===----------------------------------------------------------------------===
# Execution: jobs run on the hart itself.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT execute.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o execute.o
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS execute.o)

# ===----------------------------------------------------------------------===
# Compiler: the acc dialect and the framework passes with the Balls of the accelerator.
# ===----------------------------------------------------------------------===
set(BUCKYBALL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/ConvertLinalgToAcc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerActivation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerQuant.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerNorm.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerDecode.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/ConvertLinalgToHart.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineOperators.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-hart/OutlineReductions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert-linalg-to-acc/LowerRowOperators.cpp")

add_subdirectory("${DESIGN_REPO}/stack/compiler/plugin" plugin)
find_library(MATH_TRANSFORMS MLIRMathTransforms
  PATHS "${DESIGN_MLIR}/../.." NO_DEFAULT_PATH REQUIRED)
add_library(hart-compiler MODULE)
target_link_libraries(hart-compiler PRIVATE buckyball-compiler "${MATH_TRANSFORMS}")
set_target_properties(hart-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
