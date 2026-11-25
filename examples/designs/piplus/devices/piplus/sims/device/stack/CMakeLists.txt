# ===----------------------------------------------------------------------===
# The software stack of the piplus device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--config piplus/devices/piplus/configs/<Config>.toml --device piplus"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own: harts build buddy-opt plugins for the host, the
# device builds RISC-V libraries.
#   thread/dense, thread/action  libthread-compiler.so, program.bin, loader.o
#   hart/control   execute.o
#   hart/dense     libhart-compiler.so, execute.o
#   hart/action    libhart-compiler.so, execute.o
#   device         runtime, runtime-baremetal, runtime-linux, kernel,
#                  runtime-program, piplus-runtime; device.cmake
#   device/convert-linalg-to-device  libdevice-compiler.so
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-device-stack NONE)
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

# The threads.
ExternalProject_Add(thread-dense
  SOURCE_DIR "${DESIGN_THREAD_STACK_dense}"
  BINARY_DIR "${DESIGN_THREAD_BUILD_dense}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DTHREAD=dense
  DEPENDS midend
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

# The control hart.
ExternalProject_Add(hart-control
  SOURCE_DIR "${DESIGN_HART_STACK_control}"
  BINARY_DIR "${DESIGN_HART_BUILD_control}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=control
  DEPENDS midend
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The dense harts, which run the dense thread.
ExternalProject_Add(hart-dense
  SOURCE_DIR "${DESIGN_HART_STACK_dense}"
  BINARY_DIR "${DESIGN_HART_BUILD_dense}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=dense
  DEPENDS midend thread-dense
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The action hart, which runs the action thread.
ExternalProject_Add(hart-action
  SOURCE_DIR "${DESIGN_HART_STACK_action}"
  BINARY_DIR "${DESIGN_HART_BUILD_action}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}" -DHART=action
  DEPENDS midend thread-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The compiler of the device program, for the host.
ExternalProject_Add(device-compiler
  SOURCE_DIR "${DESIGN_DEVICE_STACK}/convert-linalg-to-device"
  BINARY_DIR "${DESIGN_DEVICE_BUILD}/convert-linalg-to-device"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  DEPENDS midend
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
  DEPENDS midend thread-dense thread-action hart-control hart-dense hart-action
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
