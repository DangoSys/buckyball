# ===----------------------------------------------------------------------===
# The software stack of the multi-rocket device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--design multi-rocket"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: cores and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   core/rocket    execute.o
#   device         runtime, runtime-baremetal, runtime-linux, engine; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(multi-rocket-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The Rocket core.
ExternalProject_Add(core-rocket
  SOURCE_DIR "${DESIGN_CORE_STACK_rocket}"
  BINARY_DIR "${DESIGN_CORE_BUILD_rocket}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=rocket
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
  DEPENDS core-rocket
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
