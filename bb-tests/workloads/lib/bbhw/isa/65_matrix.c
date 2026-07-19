#ifndef _BB_MATRIX_H_
#define _BB_MATRIX_H_

#include "isa.h"

#define BB_MATRIX_FUNC7 65
#define BB_MATRIX_MODE_OS 0ULL
#define BB_MATRIX_MODE_WS 1ULL
#define BB_MATRIX_ACC_DIRECT 0ULL
#define BB_MATRIX_ACC_FIRST 1ULL
#define BB_MATRIX_ACC_MID 2ULL
#define BB_MATRIX_ACC_LAST 3ULL
#define BB_MATRIX_CONFIG(mode, acc) (FIELD((mode), 0, 0) | FIELD((acc), 1, 2))

#define BB_MATRIX_MNK_CONFIG(m, n, k, mode)                                    \
  (FIELD((m), 0, 11) | FIELD((n), 12, 23) | FIELD((k), 24, 35) |               \
   FIELD((mode), 36, 36))

#define bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k,      \
                           mode)                                               \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) | BB_BANK2(wr_bank_id)),  \
      BB_MATRIX_MNK_CONFIG(m, n, k, mode), BB_MATRIX_FUNC7)

#define bb_matrix_mnk(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k)           \
  bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, m, n, k,            \
                     BB_MATRIX_MODE_OS)

#define bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)       \
  bb_matrix_mnk_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter, iter, iter,   \
                     ((mode) & BB_MATRIX_MODE_WS))

#define bb_matrix_4(op1_bank_id, op2_bank_id, wr_bank_id, iter)                \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter, BB_MATRIX_MODE_OS)

#define bb_matrix_5(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)          \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)

#define bb_matrix_get_macro(_1, _2, _3, _4, _5, NAME, ...) NAME
#define bb_matrix(...)                                                         \
  bb_matrix_get_macro(__VA_ARGS__, bb_matrix_5, bb_matrix_4)(__VA_ARGS__)

#define bb_matrix_os(op1_bank_id, op2_bank_id, wr_bank_id, iter)               \
  bb_matrix_4(op1_bank_id, op2_bank_id, wr_bank_id, iter)

#define bb_matrix_ws(op1_bank_id, op2_bank_id, wr_bank_id, iter)               \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter, BB_MATRIX_MODE_WS)

#define bb_matrix_os_DIRECT(op1_bank_id, op2_bank_id, wr_bank_id, iter)        \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_OS, BB_MATRIX_ACC_DIRECT))

#define bb_matrix_os_ACC_FIRST(op1_bank_id, op2_bank_id, wr_bank_id, iter)     \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_OS, BB_MATRIX_ACC_FIRST))

#define bb_matrix_os_ACC_MID(op1_bank_id, op2_bank_id, wr_bank_id, iter)       \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_OS, BB_MATRIX_ACC_MID))

#define bb_matrix_os_ACC_LAST(op1_bank_id, op2_bank_id, wr_bank_id, iter)      \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_OS, BB_MATRIX_ACC_LAST))

#define bb_matrix_ws_DIRECT(op1_bank_id, op2_bank_id, wr_bank_id, iter)        \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_WS, BB_MATRIX_ACC_DIRECT))

#define bb_matrix_ws_ACC_FIRST(op1_bank_id, op2_bank_id, wr_bank_id, iter)     \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_WS, BB_MATRIX_ACC_FIRST))

#define bb_matrix_ws_ACC_MID(op1_bank_id, op2_bank_id, wr_bank_id, iter)       \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_WS, BB_MATRIX_ACC_MID))

#define bb_matrix_ws_ACC_LAST(op1_bank_id, op2_bank_id, wr_bank_id, iter)      \
  bb_matrix_mode(op1_bank_id, op2_bank_id, wr_bank_id, iter,                   \
                 BB_MATRIX_CONFIG(BB_MATRIX_MODE_WS, BB_MATRIX_ACC_LAST))

#endif // _BB_MATRIX_H_
