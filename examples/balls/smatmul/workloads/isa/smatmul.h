#ifndef _BB_SMATMUL_H_
#define _BB_SMATMUL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

#define BB_SMATMUL_CFG(rows, cols, k)                                          \
  (FIELD((rows), 0, 11) | FIELD((cols), 12, 23) | FIELD((k), 24, 35))

#define BB_SMATMUL_RS1(op1_bank_id, op2_bank_id, wr_bank_id)                   \
  (BB_BANK0(op1_bank_id) | BB_BANK1(op2_bank_id) | BB_BANK2(wr_bank_id) |      \
   BB_ITER(0))

#define bb_smatmul_os(op1_bank_id, op2_bank_id, wr_bank_id, rows, cols, k)     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_SMATMUL_RS1(op1_bank_id, op2_bank_id, wr_bank_id),                    \
      BB_SMATMUL_CFG(rows, cols, k), BB_FUNC7(SMATMUL_OS))

#define bb_smatmul_ws(op1_bank_id, op2_bank_id, wr_bank_id, rows, cols, k)     \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_SMATMUL_RS1(op1_bank_id, op2_bank_id, wr_bank_id),                    \
      BB_SMATMUL_CFG(rows, cols, k), BB_FUNC7(SMATMUL_WS))

#define bb_smatmul_compiled(op1_bank_id, op2_bank_id, wr_bank_id, rows, cols,  \
                            k)                                                 \
  do {                                                                         \
    if ((((unsigned)(rows) + 15u) / 16u) >= 2u)                                \
      bb_smatmul_ws(op1_bank_id, op2_bank_id, wr_bank_id, rows, cols, k);      \
    else                                                                       \
      bb_smatmul_os(op1_bank_id, op2_bank_id, wr_bank_id, rows, cols, k);      \
  } while (0)

#endif // _BB_SMATMUL_H_
