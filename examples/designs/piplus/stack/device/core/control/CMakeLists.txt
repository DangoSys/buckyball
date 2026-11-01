# ===----------------------------------------------------------------------===
# The piplus control core: a CPU core without hardware threads; jobs run on
# the core itself.
#
# Built by the stack top of each simulation entry, configured with
# DESIGN_CMAKE and CORE. Products:
#   execute.o  runs jobs on the core itself; links into the device runtime
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-control-core LANGUAGES NONE)
include("${DESIGN_CMAKE}")

set(RUNTIME "${DESIGN_REPO}/stack/runtime")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

add_custom_command(
  OUTPUT execute.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_DEVICE_CORES_${DESIGN_CORE_HART_${CORE}}_SIGNATURE}ULL" "-I${RUNTIME}/scheduler"
    -c "${RUNTIME}/execute/direct.c" -o execute.o
  DEPENDS "${RUNTIME}/execute/direct.c" "${RUNTIME}/scheduler/runtime.h"
  VERBATIM)

add_custom_target(execute ALL DEPENDS execute.o)
