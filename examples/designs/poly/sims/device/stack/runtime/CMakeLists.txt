# ===----------------------------------------------------------------------===
# The poly device: its cores, their hardware threads and the runtime that
# schedules jobs onto the cores.
#
#   bbdev stack --build "--design poly"
#
# builds every thread and core stack of the device first, then configures this
# with DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the cores
#   engine                                     hardware threads and instruction streams
#   bindings                                   how each core runs jobs and each
#                                              hardware thread is loaded
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device LANGUAGES C CXX)
include("${DESIGN_CMAKE}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# How each core runs jobs, and how each hardware thread is loaded. Nothing
# references these bindings, so they are linked as objects rather than taken
# from an archive; as link options they come before the libraries they call.
add_library(bindings INTERFACE)
foreach(CORE IN LISTS DESIGN_CORES)
  target_link_options(bindings INTERFACE "${DESIGN_CORE_BUILD_${CORE}}/execute.o")
endforeach()
foreach(THREAD IN LISTS DESIGN_THREADS)
  target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_${THREAD}}/loader.o")
endforeach()
# DMA helpers are sized by the target with the fewest bank lines.
list(GET DESIGN_RUNTIME_TARGETS 0 PARAMS_TARGET)
set(RUNTIME_PARAMS "${DESIGN_ISA_DIR}/${PARAMS_TARGET}")

add_subdirectory("${DESIGN_REPO}/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)

export(TARGETS bindings runtime runtime-baremetal runtime-linux engine
  runtime-shared runtime-linux-shared engine-shared FILE device.cmake)
