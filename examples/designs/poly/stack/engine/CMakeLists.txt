# Core execution binding; instruction streams use the framework engine.
set(RUNTIME "${DESIGN_REPO}/stack/runtime")
set(ENGINE "${DESIGN_REPO}/stack/engine")
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/execute.o"
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "-DCORE_SIGNATURE=${DESIGN_CORE_SIGNATURE_${CORE}}ULL" "-I${RUNTIME}/scheduler"
    "-I${ENGINE}/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" -o "${CMAKE_BINARY_DIR}/execute.o"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/execute.c" "${RUNTIME}/scheduler/runtime.h"
    "${ENGINE}/include/engine.h"
  VERBATIM)
add_custom_target(execute ALL DEPENDS "${CMAKE_BINARY_DIR}/execute.o")
