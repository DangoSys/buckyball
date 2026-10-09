#ifndef F32ADD_H
#define F32ADD_H
#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// Four IEEE FP32 elements per 128-bit row, RNE, canonical NaN.
// rs2[4:0] selects the input group; bit 5 starts with +0 instead of reading b.
static inline void bb_f32add(uint32_t a, uint32_t b, uint32_t c, uint32_t rows,
                             uint32_t group, uint32_t first) {
  if (a > 1023 || b > 1023 || c > 1023 || !rows || group > 31 || first > 1)
    __builtin_trap();
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) |
                                BB_ITER(rows),
                            group | (uint64_t)first << 5, BB_FUNC7(F32ADD));
}
#endif
