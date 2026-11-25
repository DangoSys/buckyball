// mxint8quant of columns on the dense thread alone: the FP32 matrix X[32][4],
// one bank row a row from bank row 0, quantized along its 32 rows: output
// row c, from bank row 0 of another bank, is column c of X. Each element is
// an integer code times a power of two its column shares, the column maximum
// 100: every code is exact, and the block exponent is that power.
#include <bbhw/isa/isa.h>
#include <isa/mxint8quant.h>
#include <stdint.h>

enum { COLUMNS = 4, K = 32, OUT = MXINT8_ROWS(K) * 16 };
#define X ((volatile uint32_t *)HOST_TSS_BASE)
#define Q ((volatile uint8_t *)(HOST_TSS_BASE + K * 16))

static int code(int c, int i) {
  return i == c ? (c % 2 ? -100 : 100) : (c * 11 + i * 5) % 41 - 20;
}
static int shift(int c) { return c * 2 - 3; }

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
  for (int i = 0; i < K; ++i)
    for (int c = 0; c < COLUMNS; ++c)
      X[i * 4 + c] = scaled(code(c, i), shift(c));
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)X, 1, K, 1);
  bb_mxint8quant(1, 2, COLUMNS, K, 0, 0, 1, 1, 0);
  bb_mvout((uintptr_t)Q, 2, COLUMNS * MXINT8_ROWS(K), 1);
  asm volatile("fence" ::: "memory");
  for (int c = 0; c < COLUMNS; ++c) {
    volatile uint8_t *row = Q + c * OUT;
    for (int i = 0; i < K; ++i)
      if (row[i] != (uint8_t)code(c, i))
        return 1 + c * K + i;
    if (row[K] != 127 + shift(c))
      return 200 + c;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
