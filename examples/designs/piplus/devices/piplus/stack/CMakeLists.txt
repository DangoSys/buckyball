# ===----------------------------------------------------------------------===
# The piplus device: its harts and the runtime that schedules jobs onto them.
#
#   bbdev stack --build "--config piplus/devices/piplus/configs/<Config>.toml --device piplus"
#
# builds every hart stack of the device first, then configures this with
# DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the harts
#   kernel                                     hardware threads and instruction streams
#   bindings                                   how each hart runs jobs and
#                                              each thread is loaded
#   runtime-program                            the device program: jobs on the harts
#   piplus-runtime                             values between the harts as
#                                              copies over the coherent DDR
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(piplus-device LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# How each hart runs jobs. Nothing references these bindings, so they are
# linked as objects rather than taken from an archive; as link options they
# come before the libraries they call.
add_library(bindings INTERFACE)
target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_control}/execute.o")
target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_dense}/execute.o")
target_link_options(bindings INTERFACE "${DESIGN_HART_BUILD_action}/execute.o")
target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_dense}/loader.o")
target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_action}/loader.o")

add_subdirectory("${DESIGN_REPO}/stack/kernel" kernel)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)

# The runtime of the device program (runtime/device.c): transfers as copies
# over the coherent DDR, on the device program runtime of the framework. Its
# compiler is convert-linalg-to-device/, a project of its own for the host.
add_library(piplus-runtime STATIC runtime/device.c)
target_include_directories(piplus-runtime
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>"
)
target_compile_options(piplus-runtime PRIVATE -O2 -ffreestanding -msmall-data-limit=0)
target_link_libraries(piplus-runtime PUBLIC runtime-program)

add_library(piplus-runtime-linux STATIC runtime/device.c)
target_include_directories(piplus-runtime-linux
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>")
target_compile_options(piplus-runtime-linux PRIVATE -O2)
target_link_libraries(piplus-runtime-linux PUBLIC runtime-program-linux)

export(TARGETS bindings runtime runtime-baremetal runtime-linux kernel
  runtime-linux-shared kernel-linux kernel-shared runtime-program piplus-runtime
  runtime-program-linux piplus-runtime-linux
  FILE device.cmake)
