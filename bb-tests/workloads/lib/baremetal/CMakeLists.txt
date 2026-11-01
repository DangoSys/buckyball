# ===----------------------------------------------------------------------===
# The C API for baremetal harts. Exports bbsw-baremetal.cmake.
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.26)
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_C_COMPILER riscv64-unknown-elf-gcc)
set(CMAKE_ASM_COMPILER riscv64-unknown-elf-gcc)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
project(bbsw-baremetal C ASM)
include("${DESIGN_CMAKE}")
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH LIB)
add_compile_options(-O2 -g -march=rv64gc -mabi=lp64d -mcmodel=medany
  -msmall-data-limit=0 -fno-common)

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
  add_library(bbsw-thread-${PROFILE}-baremetal STATIC
    "${LIB}/buckyball.c"
    "${LIB}/bbhw/dma/dma.c")
  target_include_directories(bbsw-thread-${PROFILE}-baremetal
    PUBLIC "${LIB}"
           "${LIB}/bbhw/isa"
           "${LIB}/bbhw/dma"
           "${LIB}/bbsw/thread"
           ${BALL_WORKLOADS})
  target_compile_definitions(bbsw-thread-${PROFILE}-baremetal
    PUBLIC ${DESIGN_TARGET_DEFS_${PROFILE}})
  list(APPEND THREAD_LIBRARIES bbsw-thread-${PROFILE}-baremetal)
endforeach()

# RVV kernel images of the GP domain: <rvv/math.h> and the image layout.
add_library(bbsw-thread-rvv INTERFACE)
target_include_directories(bbsw-thread-rvv
  INTERFACE "${LIB}/bbsw/thread")
target_link_options(bbsw-thread-rvv
  INTERFACE "--script=${LIB}/bbsw/thread/rvv/kernel.ld")

# ===----------------------------------------------------------------------===
# Core: CoreLink between cores.
# ===----------------------------------------------------------------------===
add_library(bbsw-core-baremetal STATIC "${LIB}/bbsw/core/corelink.c")
target_include_directories(bbsw-core-baremetal
  PUBLIC "${LIB}/bbsw/core")

# ===----------------------------------------------------------------------===
# Device: its cores and harts, the workspace heap, and the boot of its harts.
# ===----------------------------------------------------------------------===
add_library(bbsw-device-baremetal STATIC "${LIB}/bbsw/device/workspace.c")
target_include_directories(bbsw-device-baremetal
  PUBLIC "${LIB}/bbsw/device")
target_compile_definitions(bbsw-device-baremetal
  PUBLIC ${DESIGN_DEFS})

# Boot of one hart of a core only; the others park.
set(CRT0_LIBRARIES)
foreach(HART IN LISTS DESIGN_DEVICE_CORES)
  add_library(bbsw-crt0-hart${HART}-baremetal OBJECT
    "${LIB}/bbsw/device/baremetal/crt0.S")
  target_compile_definitions(bbsw-crt0-hart${HART}-baremetal
    PRIVATE "CORE_HART_ID=${HART}")
  target_link_options(bbsw-crt0-hart${HART}-baremetal
    INTERFACE "-Wl,-T,${LIB}/bbsw/device/baremetal/bbsim.ld")
  list(APPEND CRT0_LIBRARIES bbsw-crt0-hart${HART}-baremetal)
endforeach()

# Boot of every hart; the reporting hart publishes the result of main.
add_library(bbsw-start-baremetal OBJECT "${LIB}/bbsw/device/baremetal/start.S")
target_compile_definitions(bbsw-start-baremetal
  PRIVATE ${DESIGN_DEFS})
target_link_options(bbsw-start-baremetal
  INTERFACE "-Wl,-T,${LIB}/bbsw/device/baremetal/bbsim.ld")

# Output through the UART and exit through the exit slot of the hart.
add_library(bbsw-syscalls-baremetal OBJECT
  "${LIB}/bbsw/device/baremetal/syscalls.c")

export(TARGETS
    ${THREAD_LIBRARIES}
    ${CRT0_LIBRARIES}
    bbsw-thread-rvv
    bbsw-core-baremetal
    bbsw-device-baremetal
    bbsw-start-baremetal
    bbsw-syscalls-baremetal
  FILE "${CMAKE_BINARY_DIR}/bbsw-baremetal.cmake")
