#ifndef BUCKYBALL_DMA_STRIDED_H
#define BUCKYBALL_DMA_STRIDED_H

#include "bb_isa.h"
#include <assert.h>

// A block holds 2^block_log2 consecutive 16-byte rows. Row i accesses DDR
// address + (i >> block_log2)*memory_stride + (i & block_mask)*16 and bank
// first + (i >> block_log2)*bank_stride + (i & block_mask), in one group.
// A one-row block permits partial rows; reads zero the unused bytes and
// writes preserve DDR outside the valid span. Larger blocks contain full
// rows, have disjoint DDR/bank spans and require rows divisible by block size.
// rs2: address[38:0], byte stride[50:39], valid bytes minus one[54:51],
// group[59:55], block_log2[62:60] in 0..4; bit 63 is reserved. rs1 uses
// unused bank fields for first and bank_stride-1; ITER counts 16-byte rows.
static inline void bb_mvin_strided(uintptr_t address, uint32_t bank,
                                   uint32_t group, uint32_t first,
                                   uint32_t bank_stride, uint32_t rows,
                                   uint32_t memory_stride, uint32_t bytes,
                                   uint32_t block_log2) {
  assert(address < (UINT64_C(1) << 39) && bank < 512 && group < 32);
  assert(first < 1024 && bank_stride >= 1 && bank_stride <= 1024 && rows > 0);
  assert(memory_stride >= 1 && memory_stride < 4096 && bytes >= 1 &&
         bytes <= 16);
  assert(block_log2 <= 4 && (rows & ((1U << block_log2) - 1)) == 0);
  assert(!block_log2 || (bytes == 16 && bank_stride >= (1U << block_log2) &&
                         memory_stride >= (16U << block_log2)));
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(first) | BB_BANK1(bank_stride - 1) | BB_BANK2(bank) |
          BB_ITER(rows) | BB_UPDATE_WR,
      FIELD(address, 0, 38) | FIELD(memory_stride, 39, 50) |
          FIELD(bytes - 1, 51, 54) | FIELD(group, 55, 59) |
          FIELD(block_log2, 60, 62),
      BB_MVIN_STRIDED_FUNC7);
}

static inline void bb_mvout_strided(uintptr_t address, uint32_t bank,
                                    uint32_t group, uint32_t first,
                                    uint32_t bank_stride, uint32_t rows,
                                    uint32_t memory_stride, uint32_t bytes,
                                    uint32_t block_log2) {
  assert(address < (UINT64_C(1) << 39) && bank < 1024 && group < 32);
  assert(first < 1024 && bank_stride >= 1 && bank_stride <= 512 && rows > 0);
  assert(memory_stride >= 1 && memory_stride < 4096 && bytes >= 1 &&
         bytes <= 16);
  assert(block_log2 <= 4 && (rows & ((1U << block_log2) - 1)) == 0);
  assert(!block_log2 || (bytes == 16 && bank_stride >= (1U << block_log2) &&
                         memory_stride >= (16U << block_log2)));
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(bank) | BB_BANK1(first) | BB_BANK2(bank_stride - 1) |
          BB_ITER(rows),
      FIELD(address, 0, 38) | FIELD(memory_stride, 39, 50) |
          FIELD(bytes - 1, 51, 54) | FIELD(group, 55, 59) |
          FIELD(block_log2, 60, 62),
      BB_MVOUT_STRIDED_FUNC7);
}
#endif
