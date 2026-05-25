#ifndef _BB_MVIN_MMIO_H_
#define _BB_MVIN_MMIO_H_

#include "isa.h"

#define BB_MVIN_MMIO_FUNC7 35

// mvin_mmio: DMA-load metadata from DRAM into MMIO SRAM.
// DMA transfers full 128-bit rows (16 bytes each); `col` controls the per-row
// byte mask written into MMIO (first `col` bytes written, rest zero-masked).
//
//   dram_addr  : source DRAM virtual address (39-bit)
//   mmio_addr  : destination MMIO byte address (17-bit)
//   row        : number of 128-bit rows to load (uses BB_ITER)
//   col        : valid bytes per row (1..16); rest is mask-zeroed in MMIO
//
// rs1 (dependencies):
//   [9:0]   = 0 (no main bank dependency)
//   [63:30] = row (BB_ITER, also serves as DMA beat count)
//
// rs2 (logic):
//   [38:0]  = dram_addr
//   [55:39] = mmio_addr (17-bit byte address)
//   [63:56] = col (write mask: valid bytes per row, 1..16)
//
// funct7 = 0x23 (35 decimal)
#define bb_mvin_mmio(dram_addr, mmio_addr, row, col)                           \
  BUCKYBALL_INSTRUCTION_R_R(BB_ITER(row),                                      \
                            (FIELD(dram_addr, 0, 38) |                         \
                             FIELD(mmio_addr, 39, 55) | FIELD(col, 56, 63)),   \
                            BB_MVIN_MMIO_FUNC7)

#endif // _BB_MVIN_MMIO_H_
