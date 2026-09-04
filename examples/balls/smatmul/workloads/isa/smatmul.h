#ifndef _BB_SMATMUL_H_
#define _BB_SMATMUL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>
#define BB_SMATMUL_CFG(rows, cols, first, last, output_base)                   \
  (FIELD((rows), 0, 11) | FIELD((cols), 12, 23) | FIELD((first), 24, 24) |     \
   FIELD((last), 25, 25) | FIELD((output_base), 26, 31))

#define bb_smatmul_bias(bias_bank, input_base)                                 \
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(bias_bank) | BB_ITER(4),                  \
                            FIELD((input_base), 0, 5), BB_FUNC7(SMATMUL_BIAS))

#define bb_smatmul_os(a_bank, b_bank, c_bank, rows, cols, k, first, last,      \
                      output_base)                                             \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(a_bank) | BB_BANK1(b_bank) | BB_BANK2(c_bank) | BB_ITER(k),     \
      BB_SMATMUL_CFG(rows, cols, first, last, output_base),                    \
      BB_FUNC7(SMATMUL_OS))

#endif // _BB_SMATMUL_H_
