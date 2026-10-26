# ===----------------------------------------------------------------------===
# The software stack of the toy device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--design toy"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: cores and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   core/toy       libtoy-compiler.so, execute.o
#   device         runtime, runtime-baremetal, runtime-linux, engine; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(toy-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# Toy's CPU core with its Ball accelerator.
ExternalProject_Add(core-toy
  SOURCE_DIR "${DESIGN_CORE_STACK_toy}"
  BINARY_DIR "${DESIGN_CORE_BUILD_toy}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=toy
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
  DEPENDS core-toy
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
