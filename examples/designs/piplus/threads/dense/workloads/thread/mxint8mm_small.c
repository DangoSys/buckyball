// mxint8mm on the dense thread alone: C[3,8] = A[3,64] . B[8,64]^T, A from
// bank row 1 of its bank, B from bank row 0 of its own, C into bank row 3
// with its rows 2 bank rows apart. A row's (B column's) block exponent is the
// same for both its blocks, so C is the integer dot product scaled by a power
// of two: exact in FP32, and its bits are known without an FPU.
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdint.h>

enum { M = 3, N = 8, K = 64, ROW = MXINT8_ROWS(K) * 16, PITCH = N / 4 };
#define A ((volatile uint8_t *)HOST_TSS_BASE)
#define B ((volatile uint8_t *)(HOST_TSS_BASE + 16 + M * ROW))
#define C ((volatile uint32_t *)(HOST_TSS_BASE + 16 + M * ROW + N * ROW))

static int code_a(int m, int i) { return (m * 7 + i * 3) % 31 - 15; }
static int code_b(int n, int i) { return (n * 5 + i * 11) % 29 - 14; }
static int exp_a(int m) { return 127 + m % 3 - 1; }
static int exp_b(int n) { return 127 - n % 3; }

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
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)A, 1, 1 + M * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)B, 2, N * MXINT8_ROWS(K), 1);
  bb_mxint8mm(1, 2, 3, M, N, K, 1, PITCH, 3, 0, 0, 0);
  bb_mvout((uintptr_t)C, 3, 3 + M * PITCH, 1);
  asm volatile("fence" ::: "memory");
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (int i = 0; i < K; ++i)
        sum += code_a(m, i) * code_b(n, i);
      if (C[12 + m * PITCH * 4 + n] != scaled(sum, exp_a(m) + exp_b(n) - 254))
        return 1 + m * N + n;
    }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
