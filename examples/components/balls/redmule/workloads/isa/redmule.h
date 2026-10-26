#ifndef REDMULE_H
#define REDMULE_H
#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// FP16 A[M,K], B[K,8], C[M,8]; ordered FP16 FMA, RNE, canonical NaN.
// Bases are bank rows. First starts with positive zero; every call writes C.
static inline void bb_redmule(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                              uint32_t k, uint32_t op1_base, uint32_t op2_base,
                              uint32_t wr_base, uint32_t first) {
  if (a > 1023 || b > 1023 || c > 1023 || a == b || a == c || b == c || !m ||
      m > 4095 || !k || k % 8 || op1_base > 1023 || op2_base > 1023 ||
      wr_base > 1023 || first > 1)
    __builtin_trap();
  uint64_t special = (uint64_t)m | (uint64_t)op1_base << 12 |
                     (uint64_t)op2_base << 22 | (uint64_t)wr_base << 32 |
                     (uint64_t)first << 42;
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) |
                                BB_ITER(k),
                            special, BB_FUNC7(REDMULE_F16));
}
#endif
