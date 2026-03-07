#ifndef _BB_GEMMINI_COMPUTE_ACCUMULATED_H_
#define _BB_GEMMINI_COMPUTE_ACCUMULATED_H_

#include "isa.h"

#define BB_GEMMINI_COMPUTE_ACCUMULATED_FUNC7 45

// Compute matmul reusing previously accumulated results
// op1_bank_id: bank for A matrix
// op2_bank_id: bank for B/D matrix
// wr_bank_id: bank for C output
// iter: number of rows
#define bb_gemmini_compute_accumulated(op1_bank_id, op2_bank_id, wr_bank_id,   \
                                       iter)                                   \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                             BB_BANK2(wr_bank_id) | BB_RD0 | BB_RD1 | BB_WR),  \
                            (FIELD(iter, 0, 9)),                               \
                            BB_GEMMINI_COMPUTE_ACCUMULATED_FUNC7)

#endif // _BB_GEMMINI_COMPUTE_ACCUMULATED_H_
