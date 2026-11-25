# ===----------------------------------------------------------------------===
# The software stack of the piplus action hart simulation: the action hart
# and its threads alone in the framework's single-hart system
# (stack/sims/hart).
#
#   bbdev stack --build "--config piplus/devices/piplus/configs/<Config>.toml --hart action"
#
# configures this with DESIGN_CMAKE, generated from design.pb.
#   midend          libbuckyball-midend.so with the Balls of the threads
#   thread/action   the action thread's compiler plugin and programs
#   hart/action     the action hart's compiler plugin and execute.o
#   system          runtime, runtime-baremetal, runtime-linux, kernel; device.cmake
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-action-hart-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

ExternalProject_Add(midend
  SOURCE_DIR "${DESIGN_REPO}/stack/compiler/midend"
  BINARY_DIR "${DESIGN_MIDEND_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(thread-action
  SOURCE_DIR "${DESIGN_THREAD_STACK_action}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_action}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=action
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(hart-action
  SOURCE_DIR "${DESIGN_HART_STACK_action}"
  BINARY_DIR "${DESIGN_HART_BUILD_action}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=action
  DEPENDS midend thread-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(system
  SOURCE_DIR "${DESIGN_REPO}/stack/sims/hart"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
    -DCMAKE_C_COMPILER=riscv64-unknown-linux-gnu-gcc
    -DCMAKE_CXX_COMPILER=riscv64-unknown-linux-gnu-g++
  DEPENDS midend hart-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
