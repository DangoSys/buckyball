# ===----------------------------------------------------------------------===
# The poly device: its harts, their hardware threads and the runtime that
# schedules jobs onto the harts.
#
#   bbdev stack --build "--config poly/devices/poly/configs/<Config>.toml --device poly"
#
# builds every thread and hart stack of the device first, then configures this
# with DESIGN_CMAKE, generated from design.pb, and the RISC-V Linux toolchain.
# Products, exported in device.cmake for the programs that run on the device:
#   runtime, runtime-baremetal, runtime-linux  job scheduling over the harts
#   kernel                                     hardware threads and instruction streams
#   bindings                                   how each hart runs jobs and each
#                                              hardware thread is loaded
#   runtime-program                            the framework's device program executor
#                                              (coherent DDR), unused by poly programs
#   poly-runtime                               the device program executor of poly: entries on
#                                              the harts, transfers through the device SHM
# Its compiler, convert-linalg-to-device/, is a project of its own for the host.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(poly-device LANGUAGES C CXX)
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
# DMA helpers are sized by the target with the fewest bank lines.
list(GET DESIGN_RUNTIME_TARGETS 0 PARAMS_TARGET)
set(RUNTIME_DEFS ${DESIGN_TARGET_DEFS_${PARAMS_TARGET}})

add_subdirectory("${DESIGN_REPO}/stack/kernel" kernel)
add_subdirectory("${DESIGN_REPO}/stack/runtime" runtime)

# The executor of the device program (runtime/device.c): the entries of a
# static program on the harts, the values between them through the device
# SHM. It implements the program contract of the framework
# (stack/runtime/program/program.h) for poly.
list(TRANSFORM DESIGN_DEFS PREPEND "-D" OUTPUT_VARIABLE POLY_RUNTIME_DEFINES)
add_library(poly-runtime STATIC runtime/device.c)
target_include_directories(poly-runtime
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>"
         "$<BUILD_INTERFACE:${DESIGN_REPO}/stack/runtime/program>"
)
target_compile_options(poly-runtime PRIVATE -O2 -ffreestanding -ftls-model=local-exec -msmall-data-limit=0
  ${POLY_RUNTIME_DEFINES} "-DDEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}")
target_link_libraries(poly-runtime PUBLIC runtime-baremetal)

add_library(poly-runtime-linux STATIC runtime/device.c)
target_include_directories(poly-runtime-linux
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/runtime>"
         "$<BUILD_INTERFACE:${DESIGN_REPO}/stack/runtime/program>"
)
target_compile_definitions(poly-runtime-linux
  PRIVATE PROGRAM_LINUX "DEVICE_CONTROL_HART=${DESIGN_WORKLOAD_CONTROL_HART}")
target_compile_options(poly-runtime-linux PRIVATE -O2 ${POLY_RUNTIME_DEFINES})
target_link_libraries(poly-runtime-linux PUBLIC runtime-linux-shared)

export(TARGETS bindings runtime runtime-baremetal runtime-linux kernel
  runtime-linux-shared kernel-linux kernel-shared runtime-program runtime-program-linux poly-runtime poly-runtime-linux
  FILE device.cmake)

add_custom_target(program-runtime ALL DEPENDS runtime-program runtime-program-linux)
