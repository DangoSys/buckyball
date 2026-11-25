# ===----------------------------------------------------------------------===
# The software stack of the poly device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--config poly/devices/poly/configs/<Config>.toml --device poly"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: harts and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   thread/attention, thread/ffn  libthread-compiler.so, program.bin, loader.o
#   hart/reduce-unpack, hart/poly execute.o and their compiler plugins
#   device         runtime, runtime-baremetal, runtime-linux, kernel,
#                  runtime-program, poly-runtime; device.cmake
#   device/convert-linalg-to-device  libdevice-compiler.so
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The framework compiler with the Balls of the device: one shared library that
# every compiler plugin of the levels below links.
ExternalProject_Add(midend
  SOURCE_DIR "${DESIGN_REPO}/stack/compiler/midend"
  BINARY_DIR "${DESIGN_MIDEND_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The attention hardware thread.
if(DEFINED DESIGN_THREAD_STACK_attention)
  ExternalProject_Add(thread-attention
    SOURCE_DIR "${DESIGN_THREAD_STACK_attention}"
    BINARY_DIR "${DESIGN_THREAD_BUILD_attention}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=attention
    DEPENDS midend
    BUILD_ALWAYS ON
    INSTALL_COMMAND "")
endif()

# The ffn hardware thread.
if(DEFINED DESIGN_THREAD_STACK_ffn)
  ExternalProject_Add(thread-ffn
    SOURCE_DIR "${DESIGN_THREAD_STACK_ffn}"
    BINARY_DIR "${DESIGN_THREAD_BUILD_ffn}"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=ffn
    DEPENDS midend
    BUILD_ALWAYS ON
    INSTALL_COMMAND "")
endif()

# The reduce hart (hart 0): the reductions and the quantization between the ranks.
ExternalProject_Add(hart-reduce-unpack
  SOURCE_DIR "${DESIGN_HART_STACK_reduce-unpack}"
  BINARY_DIR "${DESIGN_HART_BUILD_reduce-unpack}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=reduce-unpack
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# Poly's CPU hart, which runs the attention and ffn threads.
ExternalProject_Add(hart-poly
  SOURCE_DIR "${DESIGN_HART_STACK_poly}"
  BINARY_DIR "${DESIGN_HART_BUILD_poly}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=poly
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The compiler of the device program, for the host.
ExternalProject_Add(device-compiler
  SOURCE_DIR "${DESIGN_DEVICE_STACK}/convert-linalg-to-device"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}/convert-linalg-to-device"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The device: the runtime that schedules jobs onto the harts.
ExternalProject_Add(device
  SOURCE_DIR "${DESIGN_DEVICE_STACK}"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++
  DEPENDS midend hart-reduce-unpack hart-poly
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

if(TARGET thread-attention)
  add_dependencies(hart-poly thread-attention)
  add_dependencies(device thread-attention)
endif()

if(TARGET thread-ffn)
  add_dependencies(hart-poly thread-ffn)
  add_dependencies(device thread-ffn)
endif()
