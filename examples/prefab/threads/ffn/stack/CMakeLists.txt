# ===----------------------------------------------------------------------===
# The ffn thread: a hardware thread that executes the Ball instruction streams
# its core issues (the engine's stream interpreter).
#
#   bbdev stack --build "--design <design>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb) and THREAD, the
# compiler target of this thread instance. Products, in the build directory:
#   lib<THREAD>-compiler.so  buddy-opt plugin: acc dialect + this thread's Balls
#   program.elf, program.bin the stream interpreter the hardware thread runs
#   loader.o                 loads program.bin and one instruction batch into a
#                            thread; links into the engine
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(ffn-thread LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(ENGINE "${DESIGN_REPO}/stack/engine")
# The banks, thread and Ball ISA of this thread, as compile definitions.
set(THREAD_DEFS ${DESIGN_TARGET_DEFS_${THREAD}})
list(TRANSFORM THREAD_DEFS PREPEND "-D")
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# ===----------------------------------------------------------------------===
# Compiler: the acc dialect and the framework passes with the Balls of this thread.
# ===----------------------------------------------------------------------===
set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${THREAD}})
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${THREAD}-compiler MODULE)
target_link_libraries(${THREAD}-compiler PRIVATE buckyball-compiler)
set_target_properties(${THREAD}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ===----------------------------------------------------------------------===
# Program: the engine's stream interpreter, freestanding at the start of the
# thread's code memory.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.elf
  COMMAND "${ELF_CC}" -O2 -march=rv64im_zicsr -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax "-Wl,-T,${ENGINE}/program/code.ld"
    "-I${ENGINE}/include" ${THREAD_DEFS}
    "${ENGINE}/program/start.S" "${ENGINE}/program/stream.c" -o program.elf
  DEPENDS "${ENGINE}/program/stream.c" "${ENGINE}/program/instructions.inc"
    "${ENGINE}/program/start.S" "${ENGINE}/program/code.ld"
  VERBATIM)
add_custom_command(
  OUTPUT program.bin
  COMMAND "${ELF_OBJCOPY}" -O binary --only-section=.text program.elf program.bin
  DEPENDS program.elf
  VERBATIM)

# ===----------------------------------------------------------------------===
# Loader: stream_prepare.c embeds program.bin and registers as THREAD_SIGNATURE.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT loader.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -ffreestanding -fno-builtin -msmall-data-limit=0
    "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}" "-I${ENGINE}/include" ${THREAD_DEFS}
    -c "${ENGINE}/program/stream_prepare.c" -o loader.o
  DEPENDS "${ENGINE}/program/stream_prepare.c" program.bin
  VERBATIM)

add_custom_target(program ALL DEPENDS program.bin loader.o)
