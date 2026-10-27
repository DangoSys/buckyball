# ===----------------------------------------------------------------------===
# The software stack of the goban device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--design goban --device <config>"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: cores and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   thread/matmul  libmatmul-compiler.so, program.bin, loader.o, libkernels*.a
#   core/goban     libengine-compiler.so, execute.o
#   device         runtime, runtime-baremetal, runtime-linux, engine; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(goban-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The matmul hardware thread.
ExternalProject_Add(thread-matmul
  SOURCE_DIR "${DESIGN_THREAD_STACK_matmul}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_matmul}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=matmul
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# Goban's CPU core, which runs the matmul thread.
ExternalProject_Add(core-goban
  SOURCE_DIR "${DESIGN_CORE_STACK_goban}"
  BINARY_DIR "${DESIGN_CORE_BUILD_goban}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=goban
  DEPENDS thread-matmul
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
  DEPENDS thread-matmul core-goban
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
