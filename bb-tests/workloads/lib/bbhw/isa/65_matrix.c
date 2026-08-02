#ifndef _BB_MATRIX_H_
#define _BB_MATRIX_H_

#include "isa.h"

#define BB_MATRIX_FUNC7 65
#define BB_MATRIX_MODE_OS 0ULL
#define BB_MATRIX_MODE_WS 1ULL

#define BB_MATRIX_MNK_CONFIG(m, n, k, mode)                                    \
  (FIELD((m), 0, 11) | FIELD((n), 12, 23) | FIELD((k), 24, 35) |               \
   FIELD((mode), 36, 36))

#define BB_MATRIX_RS1(op1_bank_id, op2_bank_id, wr_bank_id)                    \
  (BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) | BB_BANK2(wr_bank_id))

#define bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k,      \
                           mode)                                               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_MATRIX_RS1(op1_bank_id, op2_bank_id, wr_bank_id),                     \
      BB_MATRIX_MNK_CONFIG(m, n, k, mode), BB_MATRIX_FUNC7)

#define bb_matrix_mnk(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k)           \
  bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k,            \
                     BB_MATRIX_MODE_OS)

#define bb_matrix_os(op1_bank_id, op2_bank_id, wr_bank_id, dim)                \
  bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, dim, dim, dim,      \
                     BB_MATRIX_MODE_OS)

#define bb_matrix_ws(op1_bank_id, op2_bank_id, wr_bank_id, dim)                \
  bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, dim, dim, dim,      \
                     BB_MATRIX_MODE_WS)

#endif // _BB_MATRIX_H_
