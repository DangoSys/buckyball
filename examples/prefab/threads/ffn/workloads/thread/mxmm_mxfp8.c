// mxmm on the thread alone: one MXFP8 pass (first and last) of C[16,16] =
// A[16,256] . B[16,256]^T with every code 1.0 (0x38), the block exponents of
// odd A rows 2 and of odd B columns 1/2. Each result is 256 times a power of
// two, whose FP32 bits the thread knows without an FPU.
#include <bbhw/isa/isa.h>
#include <isa/mxmm.h>
#include <stdint.h>

enum { M = 16, N = 16, K = 256, ROW = K + K / 32 };
#define A ((volatile uint8_t *)HOST_TSS_BASE)
#define B ((volatile uint8_t *)(HOST_TSS_BASE + M * ROW))
#define C ((volatile uint32_t *)(HOST_TSS_BASE + M * ROW + N * ROW))

uint64_t entry(void) {
  for (int row = 0; row < M; ++row) {
    for (int i = 0; i < K; ++i)
      A[row * K + i] = 0x38;
    for (int i = 0; i < K / 32; ++i)
      A[M * K + row * (K / 32) + i] = row % 2 ? 128 : 127;
  }
  for (int col = 0; col < N; ++col) {
    for (int i = 0; i < K; ++i)
      B[col * K + i] = 0x38;
    for (int i = 0; i < K / 32; ++i)
      B[N * K + col * (K / 32) + i] = col % 2 ? 126 : 127;
  }
  for (int i = 0; i < 4 + M * N; ++i)
    C[i] = 0;
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)A, 3, M * ROW / 16, 1);
  bb_mvin((uintptr_t)B, 4, N * ROW / 16, 1);
  bb_mxmm_mxfp8(3, 4, 5, M, N, K, 1, 1, 1);
  bb_mvout((uintptr_t)C, 5, 1 + M * N / 4, 1);
  asm volatile("fence" ::: "memory");
  for (int row = 0; row < M; ++row)
    for (int col = 0; col < N; ++col)
      // 256 * 2^(row % 2) * 2^-(col % 2)
      if (C[4 + row * N + col] != (uint32_t)(127 + 8 + row % 2 - col % 2) << 23)
        return 1 + row * N + col;
  for (int bank = 3; bank <= 5; ++bank)
    bb_mem_release(bank);
  return 0;
}
