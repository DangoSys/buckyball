// intsoftmax on the thread alone: 3 rows of 10 FP32 scores into another
// bank, through a Q15 table of exp(-d / 256) (4096 entries, here halving
// every 64 steps). The table fills the TSS, so it goes into its bank first
// and the TSS then holds the rows. The expected probabilities are the Ball's
// integer definition (emu/src/softmax.rs).
#include <bbhw/isa/isa.h>
#include <isa/intsoftmax.h>
#include <stdint.h>

enum { R = 3, COLUMNS = 10, STRIDE = (COLUMNS + 3) / 4 };
#define TABLE ((volatile uint32_t *)HOST_TSS_BASE)
#define X ((volatile uint32_t *)HOST_TSS_BASE)
#define Y ((volatile uint32_t *)(HOST_TSS_BASE + R * STRIDE * 16))

static uint32_t table(int index) {
  return index ? 32768u >> (1 + index / 64) : 32768u;
}
static int score(int r, int c) { return (r * 41 + c * 67) % 97 - 48; }

// The FP32 bits of value * 2^shift, |value| < 2^24.
static uint32_t scaled(int32_t value, int shift) {
  if (!value)
    return 0;
  uint32_t sign = value < 0 ? 0x80000000u : 0;
  uint32_t magnitude = value < 0 ? -value : value;
  int top = 31;
  while (!(magnitude >> top))
    --top;
  return sign | (uint32_t)(127 + top + shift) << 23 |
         ((magnitude << (23 - top)) & 0x7fffff);
}

static int32_t q8(uint32_t bits) {
  uint32_t exponent = (bits >> 23) & 255, magnitude;
  if (exponent >= 139)
    magnitude = 1 << 20;
  else if (exponent < 118)
    magnitude = 0;
  else {
    uint32_t mantissa = (bits & 0x7fffff) | 0x800000, shift = 142 - exponent;
    uint32_t quotient = mantissa >> shift,
             remainder = mantissa & ((1u << shift) - 1);
    uint32_t half = 1u << (shift - 1);
    magnitude =
        quotient + (remainder > half || (remainder == half && (quotient & 1)));
  }
  return bits >> 31 ? -(int32_t)magnitude : (int32_t)magnitude;
}

static uint32_t f32_q15(uint32_t value) {
  if (!value)
    return 0;
  int highest = 31;
  while (!(value >> highest))
    --highest;
  return (uint32_t)(highest + 112) << 23 |
         ((value << (23 - highest)) & 0x7fffff);
}

uint64_t entry(void) {
  for (int i = 0; i < 4096; ++i)
    TABLE[i] = table(i);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)TABLE, 2, 1024, 1);
  asm volatile("fence" ::: "memory");
  // Scores of x / 16: Q8 values 16 x, the differences up to 1536 entries.
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < STRIDE * 4; ++c)
      X[r * STRIDE * 4 + c] = c < COLUMNS ? scaled(score(r, c), -4) : 0;
  bb_mvin((uintptr_t)X, 1, R * STRIDE, 1);
  bb_intsoftmax(1, 2, 3, R, COLUMNS);
  bb_mvout((uintptr_t)Y, 3, R * STRIDE, 1);
  asm volatile("fence" ::: "memory");
  for (int r = 0; r < R; ++r) {
    int32_t scores[COLUMNS], maximum = -(1 << 30);
    for (int c = 0; c < COLUMNS; ++c) {
      scores[c] = q8(scaled(score(r, c), -4));
      if (scores[c] > maximum)
        maximum = scores[c];
    }
    uint32_t weights[COLUMNS];
    uint64_t sum = 0;
    for (int c = 0; c < COLUMNS; ++c) {
      int32_t index = maximum - scores[c];
      weights[c] = table(index < 4095 ? index : 4095);
      sum += weights[c];
    }
    for (int c = 0; c < STRIDE * 4; ++c) {
      uint32_t expected =
          c < COLUMNS
              ? f32_q15(
                    (uint32_t)(((uint64_t)weights[c] * 32768 + sum / 2) / sum))
              : 0;
      if (Y[r * STRIDE * 4 + c] != expected)
        return 1 + r * 16 + c;
    }
  }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
