# ===----------------------------------------------------------------------===
# The software stack of the multi-rocket device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--config multi-rocket/devices/multi-rocket/configs/<Config>.toml --device multi-rocket"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: harts and threads build buddy-opt plugins for the
# host, the device builds RISC-V libraries.
#   hart/rocket    execute.o
#   device         runtime, runtime-baremetal, runtime-linux, kernel; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(multi-rocket-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The Rocket hart.
ExternalProject_Add(hart-rocket
  SOURCE_DIR "${DESIGN_HART_STACK_rocket}"
  BINARY_DIR "${DESIGN_HART_BUILD_rocket}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=rocket
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
  DEPENDS hart-rocket
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
