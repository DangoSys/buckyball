# ===----------------------------------------------------------------------===
# bbsw: the C API of Buckyball, the verification interface hand-written C tests
# program the hardware with. One library per level it covers, for the
# baremetal harts and for Linux programs of one simulation of a design:
#
#   cmake -G Ninja -S bb-tests/workloads/lib -B <build> -DDESIGN_CMAKE=<design.cmake>
#   ninja -C <build>
#
# Each system is a CMake build of its own, with its own compiler:
#   baremetal   riscv64-unknown-elf-gcc: libraries and boot objects
#   linux       riscv64-unknown-linux-gnu-gcc: libraries
# <build>/bbsw.cmake imports the libraries of both into the workloads that
# use them.
#
# Libraries, <system> = baremetal | linux:
#   bbsw-thread-<profile>-<system>  the instructions of the profile (base ISA,
#                                   the ISA headers of its Balls), the bank
#                                   bookkeeping of its DMA, matrix helpers
#   bbsw-core-<system>              CoreLink between cores
#   bbsw-device-<system>            the cores of the device, the workspace
#                                   heap, harts
#   bbsw-crt0-hart<hart>-baremetal  boot of one hart of a core
#   bbsw-start-baremetal            boot of every hart
#   bbsw-syscalls-baremetal         UART and exit of newlib
#   bbsw-thread-rvv                 RVV kernel images of the GP domain
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.26)
project(bbsw NONE)
include(ExternalProject)

# The libraries for baremetal harts.
ExternalProject_Add(bbsw-baremetal
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/baremetal"
  BINARY_DIR "${CMAKE_BINARY_DIR}/baremetal"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# The libraries for Linux programs.
ExternalProject_Add(bbsw-linux
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/linux"
  BINARY_DIR "${CMAKE_BINARY_DIR}/linux"
  CMAKE_GENERATOR Ninja
  CMAKE_ARGS "-DDESIGN_CMAKE=${DESIGN_CMAKE}"
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

file(WRITE "${CMAKE_BINARY_DIR}/bbsw.cmake"
  "include(\"${CMAKE_BINARY_DIR}/baremetal/bbsw-baremetal.cmake\")\n"
  "include(\"${CMAKE_BINARY_DIR}/linux/bbsw-linux.cmake\")\n")
