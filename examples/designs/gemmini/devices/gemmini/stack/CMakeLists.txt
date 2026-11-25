# ===----------------------------------------------------------------------===
# The gemmini device: Rocket harts without Balls or hardware threads, and
# the runtime that schedules jobs onto them.
#
#   bbdev stack --build "--config gemmini/devices/gemmini/configs/<Config>.toml --device gemmini"
#
# builds every hart stack of the device first, then configures this with
# DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the harts
#   kernel                                     instruction streams (no threads here)
#   bindings                                   how each hart runs jobs
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(gemmini-device LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# How each hart runs jobs. Nothing references these bindings, so they are
# linked as objects rather than taken from an archive; as link options they
# come before the libraries they call.
add_library(bindings INTERFACE)
foreach(HART IN LISTS DESIGN_HARTS)
  target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_${HART}}/execute.o")
endforeach()

add_subdirectory("${DESIGN_REPO}/stack/kernel" kernel)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)
export(TARGETS bindings runtime runtime-baremetal runtime-linux kernel
  runtime-linux-shared kernel-linux kernel-shared FILE device.cmake)
