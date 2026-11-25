cmake_minimum_required(VERSION 3.20)
project(gemmini-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# Rocket and Gemmini execute jobs through the hart runtime.
ExternalProject_Add(hart-gemmini
  SOURCE_DIR "${DESIGN_HART_STACK_gemmini}"
  BINARY_DIR "${DESIGN_HART_BUILD_gemmini}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=gemmini
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
  DEPENDS hart-gemmini
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
