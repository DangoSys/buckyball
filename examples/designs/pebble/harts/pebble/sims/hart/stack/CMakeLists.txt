# ===----------------------------------------------------------------------===
# The software stack of the pebble hart simulation: the pebble hart
# alone in the framework's single-hart system (stack/sims/hart).
#
#   bbdev stack --build "--config pebble/devices/pebble/configs/<Config>.toml --hart pebble" or "--hart pebble/harts/pebble"
#
# configures this with DESIGN_CMAKE, generated from design.pb.
#   midend          libbuckyball-midend.so with the Balls of the hart
#   hart/pebble     execute.o
#   system          runtime, runtime-baremetal, runtime-linux, kernel; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(pebble-hart-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

ExternalProject_Add(midend
  SOURCE_DIR "${DESIGN_REPO}/stack/compiler/midend"
  BINARY_DIR "${DESIGN_MIDEND_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(hart-pebble
  SOURCE_DIR "${DESIGN_HART_STACK_pebble}"
  BINARY_DIR "${DESIGN_HART_BUILD_pebble}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=pebble
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(system
  SOURCE_DIR "${DESIGN_REPO}/stack/sims/hart"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++
  DEPENDS midend hart-pebble
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
