#ifndef BUCKYBALL_FIXED_H
#define BUCKYBALL_FIXED_H
#include <assert.h>
#include <bb_isa.h>
#include <stdint.h>

// Four signed 32-bit values per bank row. Q = 2^fraction, 1 <= fraction <= 24.
// Elementwise and MEAN results round ties to even and saturate to I32.
// REQUANT/HSWISH/HSIGMOID first compute q = round(a * M[channel] /
// 2^shift[channel]). B holds sixteen M in [0, 2^30] in rows 0..3, then sixteen
// shifts 0..61 in rows 4..7. Channels repeat every four rows. HSWISH returns
// q*clamp(q+3Q,0,6Q)/(6Q); HSIGMOID returns clamp(q+3Q,0,6Q)/6.
// MUL returns a*b/Q, ADD returns a+b, RELU returns max(a,0). Their inputs
// already have scale 1/Q. MUL may broadcast the first four B rows.
// A/B/C are one-group private banks, distinct except SUM may write B. Rows are
// a nonzero multiple of four within the bank. RELU does not read B; it is still
// allocated. SUM: A has rows/4 pixels of sixteen I32 channels; B and C hold
// sixteen I64 partial sums (eight rows). Overflow is invalid. MEAN: A holds
// sixteen I64 sums, B sixteen positive I32 divisors; C receives sixteen I32
// results. MEAN requires rows=4. Fraction is unused by SUM/MEAN. Both forbid
// broadcast.
// Streaming coefficients (rs2[9]) use two B rows per A row: four M, then
// four shifts. REQUANT/HSWISH/HSIGMOID support this mode; B needs 2*rows rows.
enum {
  BB_FIXED_REQUANT = 0,
  BB_FIXED_HSWISH = 1,
  BB_FIXED_HSIGMOID = 2,
  BB_FIXED_MUL = 3,
  BB_FIXED_RELU = 4,
  BB_FIXED_ADD = 5,
  BB_FIXED_SUM = 6,
  BB_FIXED_MEAN = 7,
};
static inline void bb_fixed(uint32_t a, uint32_t b, uint32_t c, uint32_t rows,
                            uint32_t mode, uint32_t fraction,
                            uint32_t broadcast, uint32_t stream) {
  assert(mode <= BB_FIXED_MEAN && fraction >= 1 && fraction <= 24);
  assert(stream <= 1 && (!stream || mode <= BB_FIXED_HSIGMOID));
  assert(broadcast <= 1 && (!broadcast || mode == BB_FIXED_MUL));
  assert(mode != BB_FIXED_MEAN || rows == 4);
  assert(rows > 0 && rows % 4 == 0 && a != b && a != c &&
         (b != c || mode == BB_FIXED_SUM));
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) |
                                BB_ITER(rows),
                            FIELD(mode, 0, 2) | FIELD(broadcast, 3, 3) |
                                FIELD(fraction, 4, 8) | FIELD(stream, 9, 9),
                            FUNCT7(fixed, FIXED));
}
#endif
