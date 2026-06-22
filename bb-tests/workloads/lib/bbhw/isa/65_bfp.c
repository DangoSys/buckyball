#ifndef _BB_BFP_H_
#define _BB_BFP_H_

#include "isa.h"

#define BB_BFP_FUNC7 65
#define BB_BFP_MODE_OS 0ULL
#define BB_BFP_MODE_WS 1ULL
#define BB_BFP_ACC_DIRECT 0ULL
#define BB_BFP_ACC_FIRST 1ULL
#define BB_BFP_ACC_MID 2ULL
#define BB_BFP_ACC_LAST 3ULL
#define BB_BFP_CONFIG(mode, acc)                                                \
  (FIELD((mode), 0, 0) | FIELD((acc), 1, 2))

#define bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)          \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) |   \
                              BB_BANK2(wr_bank_id) | BB_ITER(iter)),            \
                             (FIELD(mode, 0, 63)), BB_BFP_FUNC7)

#define bb_BFP_4(op1_bank_id, op2_bank_id, wr_bank_id, iter)                   \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter, BB_BFP_MODE_OS)

#define bb_BFP_5(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)             \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter, mode)

#define bb_BFP_GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME
#define bb_BFP(...)                                                             \
  bb_BFP_GET_MACRO(__VA_ARGS__, bb_BFP_5, bb_BFP_4)(__VA_ARGS__)

#define bb_BFP_OS(op1_bank_id, op2_bank_id, wr_bank_id, iter)                  \
  bb_BFP_4(op1_bank_id, op2_bank_id, wr_bank_id, iter)

#define bb_BFP_WS(op1_bank_id, op2_bank_id, wr_bank_id, iter)                  \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter, BB_BFP_MODE_WS)

#define bb_BFP_OS_DIRECT(op1_bank_id, op2_bank_id, wr_bank_id, iter)           \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_OS, BB_BFP_ACC_DIRECT))

#define bb_BFP_OS_ACC_FIRST(op1_bank_id, op2_bank_id, wr_bank_id, iter)        \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_OS, BB_BFP_ACC_FIRST))

#define bb_BFP_OS_ACC_MID(op1_bank_id, op2_bank_id, wr_bank_id, iter)          \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_OS, BB_BFP_ACC_MID))

#define bb_BFP_OS_ACC_LAST(op1_bank_id, op2_bank_id, wr_bank_id, iter)         \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_OS, BB_BFP_ACC_LAST))

#define bb_BFP_WS_DIRECT(op1_bank_id, op2_bank_id, wr_bank_id, iter)           \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_WS, BB_BFP_ACC_DIRECT))

#define bb_BFP_WS_ACC_FIRST(op1_bank_id, op2_bank_id, wr_bank_id, iter)        \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_WS, BB_BFP_ACC_FIRST))

#define bb_BFP_WS_ACC_MID(op1_bank_id, op2_bank_id, wr_bank_id, iter)          \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_WS, BB_BFP_ACC_MID))

#define bb_BFP_WS_ACC_LAST(op1_bank_id, op2_bank_id, wr_bank_id, iter)         \
  bb_BFP_MODE(op1_bank_id, op2_bank_id, wr_bank_id, iter,                      \
              BB_BFP_CONFIG(BB_BFP_MODE_WS, BB_BFP_ACC_LAST))

#endif // _BB_BFP_H_
