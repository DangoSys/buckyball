# ===----------------------------------------------------------------------===
# The multi-rocket device: Rocket cores without Balls or hardware threads, and
# the runtime that schedules jobs onto them.
#
#   bbdev stack --build "--design multi-rocket --device <config>"
#
# builds every core stack of the device first, then configures this with
# DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the cores
#   engine                                     instruction streams (no threads here)
#   bindings                                   how each core runs jobs
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(multi-rocket-device LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# How each core runs jobs. Nothing references these bindings, so they are
# linked as objects rather than taken from an archive; as link options they
# come before the libraries they call.
add_library(bindings INTERFACE)
foreach(CORE IN LISTS DESIGN_CORES)
  target_link_options(bindings INTERFACE "${DESIGN_CORE_BUILD_${CORE}}/execute.o")
endforeach()
# The device has no banks, so no DMA helpers.
set(RUNTIME_DEFS)

add_subdirectory("${DESIGN_REPO}/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)
export(TARGETS bindings runtime runtime-baremetal runtime-linux engine
  runtime-shared runtime-linux-shared engine-shared FILE device.cmake)
