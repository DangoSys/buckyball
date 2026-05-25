#ifndef _BB_MMIO_SET_H_
#define _BB_MMIO_SET_H_

#include "isa.h"

#define BB_MMIO_SET_FUNC7 34

// mmio_set: bind a main bank to an MMIO region (or dealloc when size_rows=0).
//   main_bank  : main SRAM bank id (BANK0, dependency)
//   mmio_addr  : byte address into MMIO space (16-bit)
//   size_rows  : region size in 128-bit rows (8-bit, 0 = dealloc)
//
// rs1 (dependencies):
//   [9:0]   = main_bank (BANK0)
//   [63:30] = 0 (no iter)
//
// rs2 (logic):
//   [15:0]  = mmio_addr
//   [23:16] = size_rows
//
// funct7 = 0x22 (34 decimal), enable=010 (1wr, similar to mset)
#define bb_mmio_set(main_bank, mmio_addr, size_rows)                           \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(main_bank),                                                     \
      (FIELD(mmio_addr, 0, 15) | FIELD(size_rows, 16, 23)), BB_MMIO_SET_FUNC7)

#endif // _BB_MMIO_SET_H_
