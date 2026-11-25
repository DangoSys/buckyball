// mxint8quant on the dense thread alone: 2 FP32 rows of 64 to packed MXINT8
// rows, the input rows from bank row 1, 16 bank rows apart, the output from
// bank row 2 of another bank. Each element is an integer code times a power
// of two its block shares, the block maximum 100: every code is exact, and
// the block exponent is that power.
#include <bbhw/isa/isa.h>
#include <isa/mxint8quant.h>
#include <stdint.h>

enum { R = 2, K = 64, PITCH = K / 4, OUT = MXINT8_ROWS(K) * 16 };
#define X ((volatile uint32_t *)HOST_TSS_BASE)
#define Q ((volatile uint8_t *)(HOST_TSS_BASE + 16 + R * PITCH * 16))

static int code(int r, int i) {
  return i % 32 ? (r * 13 + i * 7) % 61 - 30 : 100 - r * 200;
}
static int shift(int r, int block) {
  return r ? (block ? 5 : -2) : (block ? 3 : 0);
}

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

uint64_t entry(void) {
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < K; ++i)
      X[4 + r * PITCH * 4 + i] = scaled(code(r, i), shift(r, i / 32));
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)X, 1, 1 + R * PITCH, 1);
  bb_mxint8quant(1, 2, R, K, 1, 2, 0, PITCH, 0);
  bb_mvout((uintptr_t)Q, 2, 2 + R * MXINT8_ROWS(K), 1);
  asm volatile("fence" ::: "memory");
  for (int r = 0; r < R; ++r) {
    volatile uint8_t *row = Q + 32 + r * OUT;
    for (int i = 0; i < K; ++i)
      if (row[i] != (uint8_t)code(r, i))
        return 1 + r * K + i;
    for (int block = 0; block < K / 32; ++block)
      if (row[K + block] != 127 + shift(r, block))
        return 200 + r * 2 + block;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
