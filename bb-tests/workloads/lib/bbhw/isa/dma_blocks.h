#ifndef BUCKYBALL_DMA_BLOCKS_H
#define BUCKYBALL_DMA_BLOCKS_H

#include "bb_isa.h"
#include <assert.h>

// Whole 16-byte rows in blocks of 2^block_log2 consecutive rows. Row i
// accesses DDR address + (i >> block_log2)*memory_stride*16 + (i &
// block_mask)*16 and bank first + (i >> block_log2)*bank_stride + (i &
// block_mask), in one group. The strided DMA's byte stride stops at 4095 bytes;
// this stride counts rows, up to 65535 (1 MB). Blocks have disjoint DDR and
// bank spans; rows is a multiple of the block size. rs2: address[38:0], row
// stride[54:39], group[59:55], block_log2[62:60] in 0..4; bit 63 is reserved.
// rs1 as the strided DMA: unused bank fields for first and bank_stride-1; ITER
// counts 16-byte rows.
static inline void bb_mvin_blocks(uintptr_t address, uint32_t bank,
                                  uint32_t group, uint32_t first,
                                  uint32_t bank_stride, uint32_t rows,
                                  uint32_t memory_stride, uint32_t block_log2) {
  assert(address < (UINT64_C(1) << 39) && bank < 512 && group < 32);
  assert(first < 1024 && bank_stride >= 1 && bank_stride <= 1024 && rows > 0);
  assert(block_log2 <= 4 && (rows & ((1U << block_log2) - 1)) == 0);
  assert(memory_stride < 65536 && memory_stride >= (1U << block_log2) &&
         bank_stride >= (1U << block_log2));
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(first) | BB_BANK1(bank_stride - 1) | BB_BANK2(bank) |
          BB_ITER(rows) | BB_UPDATE_WR,
      FIELD(address, 0, 38) | FIELD(memory_stride, 39, 54) |
          FIELD(group, 55, 59) | FIELD(block_log2, 60, 62),
      BB_MVIN_BLOCKS_FUNC7);
}

static inline void bb_mvout_blocks(uintptr_t address, uint32_t bank,
                                   uint32_t group, uint32_t first,
                                   uint32_t bank_stride, uint32_t rows,
                                   uint32_t memory_stride,
                                   uint32_t block_log2) {
  assert(address < (UINT64_C(1) << 39) && bank < 1024 && group < 32);
  assert(first < 1024 && bank_stride >= 1 && bank_stride <= 512 && rows > 0);
  assert(block_log2 <= 4 && (rows & ((1U << block_log2) - 1)) == 0);
  assert(memory_stride < 65536 && memory_stride >= (1U << block_log2) &&
         bank_stride >= (1U << block_log2));
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(bank) | BB_BANK1(first) | BB_BANK2(bank_stride - 1) |
          BB_ITER(rows),
      FIELD(address, 0, 38) | FIELD(memory_stride, 39, 54) |
          FIELD(group, 55, 59) | FIELD(block_log2, 60, 62),
      BB_MVOUT_BLOCKS_FUNC7);
}
#endif
