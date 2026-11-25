// mxint8mm with bias, in place, on the dense thread alone: C[3,8] = bias +
// A[3,64] . B[8,64]^T with the bias the 2 bank rows before C (bank rows 2
// and 3), C from bank row 4 with its rows 3 bank rows apart, written in
// place: the bank row each C row leaves after it keeps what was there. A
// row's (B column's) block exponent is the same for both its blocks and the
// bias is integer, so C is exact in FP32 and known without an FPU.
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdint.h>

enum { M = 3, N = 8, K = 64, ROW = MXINT8_ROWS(K) * 16, PITCH = 3, BASE = 4 };
enum { C_ROWS = BASE + M * PITCH };
#define A ((volatile uint8_t *)HOST_TSS_BASE)
#define B ((volatile uint8_t *)(HOST_TSS_BASE + 16 + M * ROW))
#define C ((volatile uint32_t *)(HOST_TSS_BASE + 16 + M * ROW + N * ROW))

static int code_a(int m, int i) { return (m * 5 + i * 7) % 29 - 14; }
static int code_b(int n, int i) { return (n * 3 + i * 13) % 31 - 15; }
static int exp_a(int m) { return 127 - m % 2; }
static int exp_b(int n) { return 127 + n % 3 - 1; }
static int bias(int n) { return n * 9 - 30; }

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
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < ROW; ++i)
      A[16 + m * ROW + i] = 0;
    for (int i = 0; i < K; ++i) {
      A[16 + m * ROW + i] = (uint8_t)code_a(m, i);
      A[16 + m * ROW + K + i / 32] = (uint8_t)exp_a(m);
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int i = 0; i < ROW; ++i)
      B[n * ROW + i] = 0;
    for (int i = 0; i < K; ++i) {
      B[n * ROW + i] = (uint8_t)code_b(n, i);
      B[n * ROW + K + i / 32] = (uint8_t)exp_b(n);
    }
  }
  // Every word of the C bank holds 0x5a5a5a5a but the bias.
  for (int i = 0; i < C_ROWS * 4; ++i)
    C[i] = 0x5a5a5a5au;
  for (int n = 0; n < N; ++n)
    C[(BASE - N / 4) * 4 + n] = scaled(bias(n), 0);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)A, 1, 1 + M * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)B, 2, N * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)C, 3, C_ROWS, 1);
  bb_mxint8mm(1, 2, 3, M, N, K, 1, PITCH, BASE, 0, 1, 1);
  bb_mvout((uintptr_t)C, 3, C_ROWS, 1);
  asm volatile("fence" ::: "memory");
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (int i = 0; i < K; ++i)
        sum += code_a(m, i) * code_b(n, i);
      // Shifts are -2..1: in units of 2^-2, bias and product are integers.
      int shift = exp_a(m) + exp_b(n) - 254;
      int32_t quarters = bias(n) * 4 + sum * (1 << (shift + 2));
      if (C[(BASE + m * PITCH) * 4 + n] != scaled(quarters, -2))
        return 1 + m * N + n;
    }
  for (int m = 0; m < M; ++m)
    for (int i = N; i < PITCH * 4; ++i)
      if (C[(BASE + m * PITCH) * 4 + i] != 0x5a5a5a5au)
        return 100 + m;
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
