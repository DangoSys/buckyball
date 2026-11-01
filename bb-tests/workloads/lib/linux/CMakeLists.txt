# ===----------------------------------------------------------------------===
# The C API for Linux programs. Exports bbsw-linux.cmake.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.26)
set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-gcc)
project(bbsw-linux C)
include("${DESIGN_CMAKE}")
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH LIB)
add_compile_options(-O2 -g -march=rv64gc)

# ===----------------------------------------------------------------------===
# Thread: the instructions of each profile and the bank bookkeeping of its DMA.
# ===----------------------------------------------------------------------===
set(THREAD_LIBRARIES)
foreach(PROFILE IN LISTS DESIGN_PROFILES)
  # The ISA headers of the Balls of the profile: <isa/<ball>.h>.
  set(BALL_WORKLOADS)
  foreach(BALL IN LISTS DESIGN_PROFILES_${PROFILE}_THREAD_BALLDOMAIN_BALLS)
    list(APPEND BALL_WORKLOADS
      "${DESIGN_REPO}/${DESIGN_PROFILES_${PROFILE}_THREAD_BALLDOMAIN_BALLS_${BALL}_SOURCE_PATH}/workloads")
  endforeach()
  add_library(bbsw-thread-${PROFILE}-linux STATIC
    "${LIB}/buckyball.c"
    "${LIB}/bbhw/dma/dma.c")
  target_include_directories(bbsw-thread-${PROFILE}-linux
    PUBLIC "${LIB}"
           "${LIB}/bbhw/isa"
           "${LIB}/bbhw/dma"
           "${LIB}/bbsw/thread"
           ${BALL_WORKLOADS})
  target_compile_definitions(bbsw-thread-${PROFILE}-linux
    PUBLIC ${DESIGN_TARGET_DEFS_${PROFILE}})
  list(APPEND THREAD_LIBRARIES bbsw-thread-${PROFILE}-linux)
endforeach()

# ===----------------------------------------------------------------------===
# Core: CoreLink between cores.
# ===----------------------------------------------------------------------===
add_library(bbsw-core-linux STATIC "${LIB}/bbsw/core/corelink.c")
target_include_directories(bbsw-core-linux
  PUBLIC "${LIB}/bbsw/core")

# ===----------------------------------------------------------------------===
# Device: its cores and the workspace heap.
# ===----------------------------------------------------------------------===
add_library(bbsw-device-linux STATIC "${LIB}/bbsw/device/workspace.c")
target_include_directories(bbsw-device-linux
  PUBLIC "${LIB}/bbsw/device")
target_compile_definitions(bbsw-device-linux
  PUBLIC ${DESIGN_DEFS})

export(TARGETS
    ${THREAD_LIBRARIES}
    bbsw-core-linux
    bbsw-device-linux
  FILE "${CMAKE_BINARY_DIR}/bbsw-linux.cmake")
