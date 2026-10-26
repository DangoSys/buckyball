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

// E4M3 codes in row-major A[M,K] and B[N,K], followed by E8M0[M,K/32]
// and E8M0[N,K/32]. C is row-major FP32. K is a multiple of 32.
#define bb_smatmul_mxfp8(a_bank, b_bank, c_bank, rows, cols, k, first, last,   \
                         output_base)                                          \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(a_bank) | BB_BANK1(b_bank) | BB_BANK2(c_bank) | BB_ITER(k),     \
      BB_SMATMUL_CFG(rows, cols, first, last, output_base),                    \
      BB_FUNC7(SMATMUL_MXFP8))

// Row-major FP32 A[M,K], B[N,K], C[M,N]. K is a multiple of four.
#define bb_smatmul_f32(a_bank, b_bank, c_bank, rows, cols, k, first, last,     \
                       output_base)                                            \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(a_bank) | BB_BANK1(b_bank) | BB_BANK2(c_bank) | BB_ITER(k),     \
      BB_SMATMUL_CFG(rows, cols, first, last, output_base),                    \
      BB_FUNC7(SMATMUL_F32))

#endif // _BB_SMATMUL_H_
