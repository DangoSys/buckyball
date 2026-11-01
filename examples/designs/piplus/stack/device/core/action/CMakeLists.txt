# ===----------------------------------------------------------------------===
# The piplus action core: a CPU that runs its action hardware thread.
#
# Built by the stack top of each simulation entry, configured with
# DESIGN_CMAKE and CORE. Products:
#   execute.o  how the core runs jobs (through the engine)
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-action-core LANGUAGES NONE)
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
