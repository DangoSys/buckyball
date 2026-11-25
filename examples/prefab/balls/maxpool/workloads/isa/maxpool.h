#ifndef _BB_MAXPOOL_H_
#define _BB_MAXPOOL_H_

#include <bb_isa.h>

// Group g of the input and output banks is channel block g (16 INT8 channels
// per row). Both banks have the same number of groups; every group is pooled
// with the same geometry, so one instruction pools all channel blocks.

#define bb_maxpool(input_bank, output_bank, input_side, output_side, kernel,   \
                   stride, padding, input_base, output_base, output_stride,    \
                   start_row, start_col)                                       \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK2(output_bank) |                           \
          BB_ITER((output_side) * (output_side)) | BB_UPDATE_WR,               \
      FIELD(input_side, 0, 3) | FIELD(output_side, 4, 7) |                     \
          FIELD(kernel, 8, 11) | FIELD(stride, 12, 15) |                       \
          FIELD(padding, 16, 19) | FIELD(input_base, 20, 25) |                 \
          FIELD(output_base, 26, 31) | FIELD(output_stride, 32, 37) |          \
          FIELD(start_row, 38, 41) | FIELD(start_col, 42, 45),                 \
      FUNCT7(maxpool, MAXPOOL))

// Input: four U32 keys per bank row, interleaved across input groups.
// Output: two [key, original-index] pairs per row in one output group.
// rs2[31:0] is N, rs2[47:32] is K; upper bits are reserved.
#define bb_topk_u32(input_bank, output_bank, rows, n, k)                       \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(input_bank) | BB_BANK2(output_bank) |     \
                                BB_ITER(rows) | BB_UPDATE_WR,                  \
                            FIELD(n, 0, 31) | FIELD(k, 32, 47),                \
                            FUNCT7(maxpool, TOPK_U32))

#endif // _BB_MAXPOOL_H_
