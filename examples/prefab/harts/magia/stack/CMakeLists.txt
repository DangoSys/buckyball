# ===----------------------------------------------------------------------===
# The MAGIA hart: Loom, an RV64GC process CPU, and the mesh of tiles it
# launches.
#
#   bbdev stack --build "--config <config> --device <device>"
#
# builds it for each design that uses it, configured with DESIGN_CMAKE
# (generated from design.pb). Products:
#   mesh.bin  the firmware of the tiles: the GEMM service
#   mesh.o    the firmware as the section .magia of a program of Loom
#   gemm.o    magia_gemm on Loom, bare metal: a product as jobs of the mesh
#   gemm-linux.o      magia_gemm on Loom under Linux
#   firmware-linux.o  the firmware as data of a Linux program of Loom
# ===----------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20)
project(magia-hart LANGUAGES NONE)
include("${DESIGN_CMAKE}")

set(STACK "${CMAKE_CURRENT_SOURCE_DIR}")
# The runtime of a tile: its start, its drivers and where it is linked.
set(TILE "${STACK}/../workloads/lib")
find_program(ELF_CC riscv64-unknown-elf-gcc REQUIRED)
find_program(ELF_OBJCOPY riscv64-unknown-elf-objcopy REQUIRED)
find_program(LINUX_CC riscv64-unknown-linux-gnu-gcc REQUIRED)
file(GLOB TILE_HEADERS CONFIGURE_DEPENDS "${TILE}/include/*.h")

add_custom_command(
  OUTPUT mesh.elf mesh.bin mesh.o
  COMMAND "${ELF_CC}"
    -march=rv32imc_zicsr_xcvelw -mabi=ilp32 -O2 -g
    -ffreestanding -fno-builtin -fno-common -fno-stack-protector
    -ffunction-sections -fdata-sections -nostdlib -nostartfiles
    -DCV32E40P -D__cv32e40p__ -DMAGIA_MAILBOX_BASE=0x60000000
    "-I${TILE}/include" "-I${STACK}/firmware" "-T${TILE}/link.ld"
    "${TILE}/crt0.S" "${TILE}/runtime.c" "${STACK}/firmware/service.c"
    -Wl,--gc-sections -o mesh.elf
  COMMAND "${ELF_OBJCOPY}" -O binary mesh.elf mesh.bin
  COMMAND "${ELF_CC}" -march=rv64gc -mabi=lp64d -c "${STACK}/firmware/mesh.S" -o mesh.o
  DEPENDS ${TILE_HEADERS} "${STACK}/firmware/gemm.h" "${STACK}/firmware/service.c"
    "${STACK}/firmware/mesh.S" "${TILE}/crt0.S" "${TILE}/runtime.c" "${TILE}/link.ld"
  VERBATIM)

add_custom_command(
  OUTPUT gemm.o
  COMMAND "${ELF_CC}" -O2 -march=rv64gc -mabi=lp64d -mcmodel=medany -msmall-data-limit=0 -mno-relax
    -c "${STACK}/runtime/mesh/gemm.c" -o gemm.o
  DEPENDS "${STACK}/runtime/mesh/gemm.c" "${STACK}/runtime/mesh/jobs.h"
  VERBATIM)

add_custom_command(
  OUTPUT gemm-linux.o
  COMMAND "${LINUX_CC}" -O2 -fPIC -march=rv64gc -mabi=lp64d
    -c "${STACK}/runtime/mesh/linux.c" -o gemm-linux.o
  DEPENDS "${STACK}/runtime/mesh/linux.c" "${STACK}/runtime/mesh/jobs.h"
  VERBATIM)

add_custom_command(
  OUTPUT firmware-linux.o
  COMMAND "${LINUX_CC}" -fPIC -march=rv64gc -mabi=lp64d -c "${STACK}/firmware/linux.S" -o firmware-linux.o
  DEPENDS "${STACK}/firmware/linux.S" mesh.bin
  VERBATIM)

add_custom_target(mesh ALL DEPENDS mesh.o gemm.o gemm-linux.o firmware-linux.o)
