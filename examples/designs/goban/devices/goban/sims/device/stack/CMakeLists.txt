# ===----------------------------------------------------------------------===
# The software stack of the goban device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--config goban/devices/goban/configs/<Config>.toml --device goban"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: harts and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   thread/matmul  program.bin, loader.o
#   hart/goban     libhart-compiler.so, execute.o
#   device         runtime, runtime-baremetal, runtime-linux, kernel; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(goban-device-stack NONE)
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

# The matmul hardware thread.
ExternalProject_Add(thread-matmul
  SOURCE_DIR "${DESIGN_THREAD_STACK_matmul}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_matmul}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=matmul
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# Goban's CPU hart, which runs the matmul thread.
ExternalProject_Add(hart-goban
  SOURCE_DIR "${DESIGN_HART_STACK_goban}"
  BINARY_DIR "${DESIGN_HART_BUILD_goban}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=goban
  DEPENDS midend thread-matmul
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
  DEPENDS midend thread-matmul hart-goban
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
