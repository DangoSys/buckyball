# ===----------------------------------------------------------------------===
# The software stack of the piplus device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--design piplus --device <config>"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: cores build buddy-opt plugins for the host, the
# device builds RISC-V libraries.
#   thread/dense, thread/action  lib<thread>-compiler.so, program.bin, loader.o
#   core/control   execute.o
#   core/dense     execute.o
#   core/action    execute.o
#   core/compiler  libcore-compiler.so
#   device         runtime, runtime-baremetal, runtime-linux, engine,
#                  runtime-program, piplus-runtime; device.cmake
#   device/compiler  libdevice-compiler.so
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The threads.
ExternalProject_Add(thread-dense
  SOURCE_DIR "${DESIGN_THREAD_STACK_dense}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_dense}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=dense
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

ExternalProject_Add(thread-action
  SOURCE_DIR "${DESIGN_THREAD_STACK_action}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_action}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The control core.
ExternalProject_Add(core-control
  SOURCE_DIR "${DESIGN_CORE_STACK_control}"
  BINARY_DIR "${DESIGN_CORE_BUILD_control}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=control
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The dense cores, which run the dense thread.
ExternalProject_Add(core-dense
  SOURCE_DIR "${DESIGN_CORE_STACK_dense}"
  BINARY_DIR "${DESIGN_CORE_BUILD_dense}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=dense
  DEPENDS thread-dense
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The action core, which runs the action thread.
ExternalProject_Add(core-action
  SOURCE_DIR "${DESIGN_CORE_STACK_action}"
  BINARY_DIR "${DESIGN_CORE_BUILD_action}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DCORE=action
  DEPENDS thread-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The compiler of the graphs on the cores, for the host.
ExternalProject_Add(core-compiler
  SOURCE_DIR "${DESIGN_DEVICE_STACK}/core/compiler"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}/core/compiler"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The compiler of the device program, for the host.
ExternalProject_Add(device-compiler
  SOURCE_DIR "${DESIGN_DEVICE_STACK}/compiler"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}/compiler"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
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
  DEPENDS thread-dense thread-action core-control core-dense core-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
