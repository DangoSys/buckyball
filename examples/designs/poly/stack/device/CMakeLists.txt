# ===----------------------------------------------------------------------===
# The poly device: its cores, their hardware threads and the runtime that
# schedules jobs onto the cores.
#
#   bbdev stack --build "--design poly --device <config>"
#
# builds every thread and core stack of the device first, then configures this
# with DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the cores
#   engine                                     hardware threads and instruction streams
#   bindings                                   how each core runs jobs and each
#                                              hardware thread is loaded
#   runtime-program                            the device program: jobs on the cores
#   poly-runtime                               values between the cores over
#                                              the core links
# Its compiler, compiler/, is a project of its own for the host.
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
foreach(THREAD IN LISTS DESIGN_THREAD_STACKS)
  target_link_options(bindings INTERFACE "${DESIGN_THREAD_BUILD_${THREAD}}/loader.o")
endforeach()
# DMA helpers are sized by the target with the fewest bank lines.
list(GET DESIGN_RUNTIME_TARGETS 0 PARAMS_TARGET)
set(RUNTIME_DEFS ${DESIGN_TARGET_DEFS_${PARAMS_TARGET}})

add_subdirectory("${DESIGN_REPO}/stack/engine" engine)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)

# The runtime of the device program (runtime/device.c): transfers over the
# core links, on the device program runtime of the framework.
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE POLY_RUNTIME_DEFINES)
add_library(poly-runtime STATIC runtime/device.c)
target_include_directories(poly-runtime
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>"
)
target_compile_options(poly-runtime PRIVATE -O2 -ffreestanding -msmall-data-limit=0
  ${POLY_RUNTIME_DEFINES})
target_link_libraries(poly-runtime PUBLIC runtime-program)

export(TARGETS bindings runtime runtime-baremetal runtime-linux engine
  runtime-linux-shared engine-linux engine-shared runtime-program poly-runtime
  FILE device.cmake)
