# ===----------------------------------------------------------------------===
# The software stack of the MAGIA device simulation: every level the device
# needs, built bottom up.
#
#   bbdev stack --build "--config magia/devices/magia/configs/<Config>.toml --device magia"
#
# configures this with DESIGN_CMAKE, generated from design.pb. Each level is a
# CMake build of its own.
#   hart/magia  mesh.bin, mesh.o, gemm.o
#   device      device.ld: where a program of the device lies (no build)
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(magia-device-stack NONE)
include(ExternalProject)
include("${DESIGN_CMAKE}")

# The MAGIA hart: the firmware of its tiles and Loom's runtime.
ExternalProject_Add(hart-magia
  SOURCE_DIR "${DESIGN_HART_STACK_magia}"
  BINARY_DIR "${DESIGN_HART_BUILD_magia}"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")
