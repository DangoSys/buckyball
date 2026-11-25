// smatmul on the thread alone: C = A * B + bias for 16x16x16 INT8, operands
// and result in the TSS, checked by the thread against its own reference.
#include <bbhw/isa/isa.h>
#include <isa/smatmul.h>
#include <stdint.h>

enum { M = 16, N = 16, K = 16 };
#define A ((volatile int8_t *)HOST_TSS_BASE)
#define B ((volatile int8_t *)(HOST_TSS_BASE + M * K))
#define BIAS ((volatile int32_t *)(HOST_TSS_BASE + M * K + K * N))
#define C ((volatile int32_t *)(HOST_TSS_BASE + 1024))

uint64_t entry(void) {
  for (int row = 0; row < M; ++row)
    for (int k = 0; k < K; ++k)
      A[row * K + k] = (row * 3 + k * 5) % 13 - 6;
  for (int k = 0; k < K; ++k)
    for (int col = 0; col < N; ++col)
      B[k * N + col] = (k * 7 + col * 2) % 11 - 5;
  for (int col = 0; col < N; ++col)
    BIAS[col] = col - 8;

  for (int bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)BIAS, 0, 4, 1);
  bb_mvin((uintptr_t)A, 1, M, 1);
  bb_mvin((uintptr_t)B, 2, K, 1);
  bb_smatmul_bias(0, 0);
  bb_smatmul_os(1, 2, 3, M, N, K, 1, 1, 0);
  bb_mvout((uintptr_t)C, 3, M * 4, 1);
  asm volatile("fence" ::: "memory");

  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col) {
      int32_t expected = BIAS[col];
      for (int k = 0; k < K; ++k)
        expected += A[row * K + k] * B[k * N + col];
      if (C[(row * 4 + col / 4) * 4 + col % 4] != expected)
        return 1 + row * N + col;
    }
  for (int bank = 0; bank < 4; ++bank)
    bb_mem_release(bank);
  return 0;
}
