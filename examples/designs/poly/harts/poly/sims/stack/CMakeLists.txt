# ===----------------------------------------------------------------------===
# The software stack of the poly poly hart simulation: the poly hart and its attention and ffn threads alone in the
# framework's single-hart system (stack/sims/hart).
#
#   bbdev stack --build "--hart poly/harts/poly"
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-poly-hart-stack NONE)
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
ExternalProject_Add(thread-attention
  SOURCE_DIR "${DESIGN_THREAD_STACK_attention}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_attention}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=attention
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(thread-ffn
  SOURCE_DIR "${DESIGN_THREAD_STACK_ffn}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_ffn}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=ffn
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(hart-poly
  SOURCE_DIR "${DESIGN_HART_STACK_poly}"
  BINARY_DIR "${DESIGN_HART_BUILD_poly}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=poly
  DEPENDS midend thread-attention thread-ffn
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(system
  SOURCE_DIR "${DESIGN_REPO}/stack/sims/hart"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++
  DEPENDS midend hart-poly
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
