# ===----------------------------------------------------------------------===
# The pebble device: its harts, their hardware threads and the runtime that
# schedules jobs onto the harts.
#
#   bbdev stack --build "--config pebble/devices/pebble/configs/<Config>.toml --device pebble"
#
# builds every thread and hart stack of the device first, then configures this
# with DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the harts
#   kernel                                     hardware threads and instruction streams
#   bindings                                   how each hart runs jobs and each
#                                              hardware thread is loaded
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(pebble-device LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# How each hart runs jobs, and how each hardware thread is loaded. Nothing
# references these bindings, so they are linked as objects rather than taken
# from an archive; as link options they come before the libraries they call.
add_library(bindings INTERFACE)
foreach(HART IN LISTS DESIGN_HARTS)
  target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_${HART}}/execute.o")
endforeach()
foreach(THREAD IN LISTS DESIGN_THREAD_STACKS)
  target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_${THREAD}}/loader.o")
endforeach()

add_subdirectory("${DESIGN_REPO}/stack/kernel" kernel)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)
export(TARGETS bindings runtime runtime-baremetal runtime-linux kernel
  runtime-linux-shared kernel-linux kernel-shared FILE device.cmake)
