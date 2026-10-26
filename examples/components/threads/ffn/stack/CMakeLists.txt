# ===----------------------------------------------------------------------===
# The ffn thread: a hardware thread that executes the Ball instruction streams
# its core issues (the engine's stream interpreter).
#
#   bbdev stack --build "--design <design>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb) and THREAD, the
# compiler target of this thread instance. Products, in the build directory:
#   lib<THREAD>-compiler.so  buddy-opt plugin: Buckyball dialect + this thread's Balls
#   program.elf, program.bin the stream interpreter the hardware thread runs
#   loader.o                 loads program.bin and one instruction batch into a
#                            thread; links into the engine
#   libkernels.a, libkernels-stream.a, kernels.rename
#                            the kernels of this thread, see Kernels below
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(ffn-thread LANGUAGES C CXX)
include("${DESIGN_CMAKE}")

set(ENGINE "${DESIGN_REPO}/stack/engine")
set(ISA "${DESIGN_ISA_DIR}/${THREAD}")
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)

# ===----------------------------------------------------------------------===
# Compiler: the Buckyball dialect and passes with the Balls of this thread.
# ===----------------------------------------------------------------------===
set(BUCKYBALL_BALLS ${DESIGN_TARGET_BALL_STACKS_${THREAD}})
set(BUCKYBALL_TARGET_TABLE "${ISA}/target.inc")
add_subdirectory("${DESIGN_REPO}/stack/compiler/midend" compiler)
add_library(${THREAD}-compiler MODULE $<TARGET_OBJECTS:buckyball-compiler>)
set_target_properties(${THREAD}-compiler PROPERTIES
  LINKER_LANGUAGE CXX LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ===----------------------------------------------------------------------===
# Program: the engine's stream interpreter, freestanding at the start of the
# thread's code memory.
# ===----------------------------------------------------------------------===
add_custom_command(
  OUTPUT program.elf
  COMMAND "${ELF_CC}" -O2 -march=rv64im -mabi=lp64 -mcmodel=medany
    -ffreestanding -fno-builtin -fno-pic -fno-stack-protector -fno-jump-tables
    -fno-asynchronous-unwind-tables -msmall-data-limit=0 -mno-relax
    -nostdlib -nostartfiles -Wl,--no-relax "-Wl,-T,${ENGINE}/program/code.ld"
    "-I${ENGINE}/include" "-I${ISA}"
    "${ENGINE}/program/start.S" "${ENGINE}/program/stream.c" -o program.elf
  DEPENDS "${ENGINE}/program/stream.c" "${ENGINE}/program/instructions.inc"
    "${ENGINE}/program/start.S" "${ENGINE}/program/code.ld" "${ISA}/params.h"
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
    "--embed-dir=${CMAKE_CURRENT_BINARY_DIR}" "-I${ENGINE}/include" "-I${ISA}"
    -c "${ENGINE}/program/stream_prepare.c" -o loader.o
  DEPENDS "${ENGINE}/program/stream_prepare.c" program.bin "${ISA}/params.h"
  VERBATIM)

add_custom_target(program ALL DEPENDS program.bin loader.o)

# ===----------------------------------------------------------------------===
# Kernels: the RVV kernels of the GP domain and the kernels of this thread's
# Balls, with the host code that launches them, built against this thread's
# ISA. Every exported symbol gets the suffix __${THREAD}, so the kernels of
# several threads link into one program; models apply kernels.rename to the
# code they compile for this thread.
#   libkernels.a         Ball instructions issued by the calling core
#   libkernels-stream.a  Ball instructions sent to the thread through the engine
# ===----------------------------------------------------------------------===
find_program(LINUX_CXX riscv64-unknown-linux-gnu-g++ REQUIRED)
find_program(LINUX_AR riscv64-unknown-linux-gnu-ar REQUIRED)
find_program(LINUX_OBJCOPY riscv64-unknown-linux-gnu-objcopy REQUIRED)
set(RVV "${DESIGN_REPO}/stack/kernels/rvv")
add_subdirectory("${RVV}" rvv)

set(KERNEL_SOURCES
  "${RVV}/dispatch.cpp"
  "${RVV}/matmul_dispatch.cpp"
  "${RVV}/cache_dispatch.cpp"
  "${RVV}/normalization_dispatch.cpp"
  "${RVV}/pointwise_dispatch.cpp"
  "${RVV}/activation_dispatch.cpp"
  "${RVV}/flash_attention_dispatch.cpp"
  "${RVV}/flash_attention_mxfp8_dispatch.cpp")
set(KERNEL_INCLUDES
  "${ISA}"
  "${DESIGN_ISA_DIR}/design"
  "${DESIGN_REPO}/bb-tests/workloads/lib"
  "${DESIGN_REPO}/stack/kernels"
  "${DESIGN_REPO}/stack/runtime/scheduler"
  "${DESIGN_REPO}/stack/runtime/services"
  "${ENGINE}/include"
  "${DESIGN_REPO}/stack/thirdparty/buddy-mlir/llvm/mlir/include/mlir/ExecutionEngine")
set(KERNEL_EMBED_DIRS "${CMAKE_CURRENT_BINARY_DIR}/rvv")
set(KERNEL_DEPENDS rvv-images)
set(KERNEL_EXPORTS
  _mlir_ciface_rvv_attention_softmax _mlir_ciface_rvv_binary _mlir_ciface_rvv_cos
  _mlir_ciface_rvv_flash_attention _mlir_ciface_rvv_flash_attention_mxfp8 _mlir_ciface_rvv_gelu
  _mlir_ciface_rvv_layernorm _mlir_ciface_rvv_layernorm_no_bias _mlir_ciface_rvv_logsumexp_softmax
  _mlir_ciface_rvv_matmul _mlir_ciface_rvv_matmul_broadcast_rhs
  _mlir_ciface_rvv_matmul_broadcast_transpose_rhs _mlir_ciface_rvv_matmul_transpose_rhs
  _mlir_ciface_rvv_mxfp8_decode _mlir_ciface_rvv_mxfp8_encode _mlir_ciface_rvv_norm
  _mlir_ciface_rvv_norm_no_weight _mlir_ciface_rvv_rope _mlir_ciface_rvv_silu _mlir_ciface_rvv_sin
  _mlir_ciface_rvv_snake _mlir_ciface_rvv_softmax _mlir_ciface_rvv_swiglu _mlir_ciface_rvv_tanh
  _mlir_ciface_rvv_unary)
# The Balls of this thread add their kernels to the lists above.
foreach(DIRECTORY IN LISTS DESIGN_TARGET_KERNELS_${THREAD})
  get_filename_component(BALL "${DIRECTORY}/../.." ABSOLUTE)
  get_filename_component(BALL "${BALL}" NAME)
  add_subdirectory("${DIRECTORY}" balls/${BALL})
endforeach()

set(RENAME "")
foreach(SYMBOL IN LISTS KERNEL_EXPORTS)
  string(APPEND RENAME "${SYMBOL} ${SYMBOL}__${THREAD}\n")
endforeach()
file(CONFIGURE OUTPUT kernels.rename CONTENT "${RENAME}")
list(TRANSFORM KERNEL_INCLUDES PREPEND "-I")
list(TRANSFORM KERNEL_EMBED_DIRS PREPEND "--embed-dir=")

foreach(LIBRARY kernels kernels-stream)
  set(DEFINES)
  if(LIBRARY STREQUAL "kernels-stream")
    set(DEFINES -DBUCKYBALL_STREAM)
  endif()
  set(OBJECTS)
  foreach(SOURCE IN LISTS KERNEL_SOURCES)
    file(RELATIVE_PATH OBJECT "${DESIGN_REPO}" "${SOURCE}.o")
    set(OBJECT "${LIBRARY}/${OBJECT}")
    add_custom_command(
      OUTPUT ${OBJECT}
      COMMAND "${LINUX_CXX}" -std=c++20 -O2 -fPIC -ffp-contract=off ${DEFINES}
        ${KERNEL_INCLUDES} ${KERNEL_EMBED_DIRS} -MD -MF ${OBJECT}.d -c "${SOURCE}" -o ${OBJECT}
      DEPENDS "${SOURCE}" ${KERNEL_DEPENDS}
      DEPFILE ${OBJECT}.d
      VERBATIM)
    list(APPEND OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/${OBJECT}")
  endforeach()
  add_custom_command(
    OUTPUT lib${LIBRARY}.a
    COMMAND "${CMAKE_COMMAND}" -E rm -f lib${LIBRARY}.a
    COMMAND "${LINUX_AR}" rcs lib${LIBRARY}.a ${OBJECTS}
    COMMAND "${LINUX_OBJCOPY}" --redefine-syms=kernels.rename lib${LIBRARY}.a
    DEPENDS ${OBJECTS} "${CMAKE_CURRENT_BINARY_DIR}/kernels.rename"
    VERBATIM)
endforeach()
add_custom_target(kernels ALL DEPENDS libkernels.a libkernels-stream.a)
