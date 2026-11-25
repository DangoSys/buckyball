// One 32-element block against 32 packed MXINT5 weight rows.
// Each weight row has two code rows, a factor row, and an exponent row.
#include <bbhw/isa/isa.h>
#include <isa/mxint5gemv.h>
#include <stdint.h>

enum { K = 32, STRIDE = 4, EXPONENT_A = 126 };
#define A ((volatile uint8_t *)HOST_TSS_BASE)
#define B ((volatile uint8_t *)(HOST_TSS_BASE + 48))
#define C ((volatile uint8_t *)(HOST_TSS_BASE + 48 + 32 * STRIDE * 16))

static int code_a(int i) { return (i * 29) % 41 - 20; }
static int code_b(int j, int i) { return (j * 7 + i * 3) % 31 - 15; }
static int factor(int j) { return 1 + (j * 17) % 137; }
static int exponent_b(int j) { return 127 + j % 3 - 1; }

static int64_t round_shift(int64_t value, int shift) {
  if (shift == 0 || shift >= 64)
    return shift ? 0 : value;
  uint64_t magnitude = value < 0 ? -(uint64_t)value : (uint64_t)value;
  uint64_t quotient = magnitude >> shift;
  uint64_t remainder = magnitude & ((UINT64_C(1) << shift) - 1);
  uint64_t half = UINT64_C(1) << (shift - 1);
  quotient += remainder > half || (remainder == half && (quotient & 1));
  return value < 0 ? -(int64_t)quotient : (int64_t)quotient;
}

uint64_t entry(void) {
  for (int i = 0; i < 48 + 32 * STRIDE * 16; ++i)
    A[i] = 0;
  for (int i = 0; i < K; ++i)
    A[i] = (uint8_t)code_a(i);
  A[32] = EXPONENT_A;
  for (int j = 0; j < 32; ++j) {
    volatile uint8_t *row = B + j * STRIDE * 16;
    for (int i = 0; i < K; ++i) {
      unsigned bit = 5 * i, encoded = (unsigned)code_b(j, i) & 31;
      row[bit / 8] |= (uint8_t)(encoded << (bit % 8));
      if (bit % 8 > 3)
        row[bit / 8 + 1] |= (uint8_t)(encoded >> (8 - bit % 8));
    }
    row[32] = factor(j);
    row[48] = exponent_b(j);
  }
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 4);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)A, 1, 3, 1);
  for (int group = 0; group < 4; ++group)
    bb_mvin_strided((uintptr_t)(B + group * 8 * STRIDE * 16), 2, group, 0, 1,
                    8 * STRIDE, 16, 16, 0);
  bb_mxint5gemv(1, 2, 3, K, 0, 2, 0, 0, 0, 1, 0);
  bb_mvout((uintptr_t)C, 3, 3, 1);
  asm volatile("fence" ::: "memory");
  int64_t products[32];
  int scales[32], next = -1000;
  for (int j = 0; j < 32; ++j) {
    int64_t sum = 0;
    for (int i = 0; i < K; ++i)
      sum += code_a(i) * code_b(j, i);
    products[j] = sum * factor(j);
    scales[j] = EXPONENT_A + exponent_b(j) - 258;
    if (products[j] && scales[j] > next)
      next = scales[j];
  }
  int64_t values[32];
  uint64_t maximum = 0;
  for (int j = 0; j < 32; ++j) {
    int shift = 28 + scales[j] - next;
    values[j] = !products[j] ? 0
                : shift >= 0 ? products[j] * ((int64_t)1 << shift)
                             : round_shift(products[j], -shift);
    uint64_t magnitude = values[j] < 0 ? -values[j] : values[j];
    if (magnitude > maximum)
      maximum = magnitude;
  }
  int scale = maximum ? next : 0, exponent = 0;
  if (maximum) {
    int top = 63;
    while (!(maximum >> top))
      --top;
    exponent = top + scale - 28 - 6;
    exponent = exponent < -127 ? -127 : exponent > 127 ? 127 : exponent;
  }
  int shift = exponent - scale + 28;
  for (int j = 0; j < 32; ++j) {
    int64_t v = values[j], sign = (v > 0) - (v < 0);
    int64_t code = shift >= 0 ? round_shift(v, shift) : v * sign;
    if (shift < 0)
      code = shift < -6 || code > (127 >> -shift) ? 127 : code << -shift;
    code = (code > 127 ? 127 : code) * (shift >= 0 || !sign ? 1 : sign);
    if (C[j] != (uint8_t)(int8_t)code)
      return 1 + j;
  }
  if (C[32] != exponent + 127)
    return 100;
  for (int bank = 1; bank <= 3; ++bank)
    bb_mem_release(bank);
  return 0;
}
