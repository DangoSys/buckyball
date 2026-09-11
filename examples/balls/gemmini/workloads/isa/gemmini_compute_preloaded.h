#ifndef _BB_GEMMINI_COMPUTE_PRELOADED_H_
#define _BB_GEMMINI_COMPUTE_PRELOADED_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define BB_GEMMINI_COMPUTE_PRELOADED_RS2 2ULL

// Compute matmul using preloaded data: C = A * B + D(preloaded)
// op1_bank_id: bank for A matrix
// op2_bank_id: bank for B matrix (OS) or D matrix (WS)
// wr_bank_id: bank for C output
// iter: number of rows
#define bb_gemmini_compute_preloaded(op1_bank_id, op2_bank_id, wr_bank_id,     \
                                     iter, op1_base, op2_base, wr_base)        \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) | BB_BANK2(wr_bank_id) |  \
       BB_ITER(iter)),                                                         \
      BB_GEMMINI_COMPUTE_PRELOADED_RS2 | ((uint64_t)(op1_base) << 6) |         \
          ((uint64_t)(op2_base) << 16) | ((uint64_t)(wr_base) << 26),          \
      BB_FUNC7(GEMMINI_COMPUTE_PRELOADED))

#endif // _BB_GEMMINI_COMPUTE_PRELOADED_H_
