# ===----------------------------------------------------------------------===
# The software stack of the poly device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--design poly --device <config>"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: cores and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   thread/attention, thread/ffn  lib<thread>-compiler.so, program.bin, loader.o,
#                                 libkernels*.a
#   core/reduce, core/poly        execute.o and their compiler plugins
#   device         runtime, runtime-baremetal, runtime-linux, engine; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The attention hardware thread.
ExternalProject_Add(thread-attention
  SOURCE_DIR "${DESIGN_THREAD_STACK_attention}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_attention}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=attention
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The ffn hardware thread.
ExternalProject_Add(thread-ffn
  SOURCE_DIR "${DESIGN_THREAD_STACK_ffn}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_ffn}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=ffn
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The reduce core.
ExternalProject_Add(core-reduce
  SOURCE_DIR "${DESIGN_CORE_STACK_reduce}"
  BINARY_DIR "${DESIGN_CORE_BUILD_reduce}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=reduce
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# Poly's CPU core, which runs the attention and ffn threads.
ExternalProject_Add(core-poly
  SOURCE_DIR "${DESIGN_CORE_STACK_poly}"
  BINARY_DIR "${DESIGN_CORE_BUILD_poly}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=poly
  DEPENDS thread-attention thread-ffn
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The device: the runtime that schedules jobs onto the cores.
ExternalProject_Add(device
  SOURCE_DIR "${DESIGN_DEVICE_STACK}"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++
  DEPENDS thread-attention thread-ffn core-reduce core-poly
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
