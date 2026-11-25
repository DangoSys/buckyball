# ===----------------------------------------------------------------------===
# The software stack of the poly reduce-unpack hart simulation: the reduce-unpack hart and its accelerator alone in the
# framework's single-hart system (stack/sims/hart).
#
#   bbdev stack --build "--hart poly/harts/reduce-unpack"
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-reduce-unpack-hart-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

ExternalProject_Add(midend
  SOURCE_DIR "${DESIGN_REPO}/stack/compiler/midend"
  BINARY_DIR "${DESIGN_MIDEND_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The attention hardware thread.
ExternalProject_Add(hart-reduce-unpack
  SOURCE_DIR "${DESIGN_HART_STACK_reduce-unpack}"
  BINARY_DIR "${DESIGN_HART_BUILD_reduce-unpack}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=reduce-unpack
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
  DEPENDS midend hart-reduce-unpack
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
