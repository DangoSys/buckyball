# ===----------------------------------------------------------------------===
# The software stack of the reduce hart simulation: the reduce hart and its accelerator alone in the
# framework's single-hart system (stack/sims/hart).
#
#   bbdev stack --build "--hart reduce"
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(reduce-hart-stack NONE)
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
ExternalProject_Add(hart-reduce
  SOURCE_DIR "${DESIGN_HART_STACK_reduce}"
  BINARY_DIR "${DESIGN_HART_BUILD_reduce}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=reduce
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
  DEPENDS midend hart-reduce
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
