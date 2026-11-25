# ===----------------------------------------------------------------------===
# The multi-rocket rocket hart (a copy of the prefab rocket hart stack): a CPU
# hart without hardware threads; jobs run on the hart.
#
#   bbdev stack --build "--config <config> --device <device>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb) and HART. Products:
#   execute.o  runs jobs on the hart itself; links into the device runtime
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(rocket-hart LANGUAGES NONE)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

add_custom_command(
  OUTPUT execute.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DHART_SIGNATURE=${DESIGN_DEVICE_HARTS_${DESIGN_HART_HART_${HART}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o execute.o
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)

add_custom_target(execute ALL DEPENDS execute.o)
