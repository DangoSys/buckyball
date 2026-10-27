#include <bbhw/isa/isa.h>
#include <isa/gemmini.h>
#include <stdint.h>

#define DIM (THREAD_MEM_BANK_WIDTH / 8)
#define M DIM
#define N (2 * DIM)
#define K (2 * DIM)
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t c[M * N] __attribute__((aligned(64)));
static int32_t expected[9][7];
_Static_assert(DIM == 16,
               "Gemmini loop tests require the configured 16x16 array");
static uint64_t descriptor[8] __attribute__((aligned(64)));

int main(void) {
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = (i + 3 * k) % 9 - 4;
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      b[k * N + j] = (2 * k + 5 * j) % 7 - 3;
  for (int i = 0; i < M * N; ++i)
    c[i] = 0x2468ace0;
  descriptor[0] =
      ((uint64_t)(M / DIM) << 32) | ((uint64_t)(N / DIM) << 16) | (K / DIM);
  descriptor[1] = (uintptr_t)a;
  descriptor[2] = (uintptr_t)b;
  descriptor[3] = 0;
  descriptor[4] = (uintptr_t)c;
  descriptor[5] = ((uint64_t)N << 32) | K;
  descriptor[6] = (uint64_t)(N * 4) << 32;
  descriptor[7] = 0;
  bb_gemmini_config(0, 0, 0, 0, 0);
  bb_mvin_mmio((uintptr_t)descriptor, 0,
               sizeof(descriptor) / (THREAD_MEM_BANK_WIDTH / 8),
               THREAD_MEM_BANK_WIDTH / 8);
  bb_gemmini_loop_ws(0, 1, 2, 1);
  bb_barrier();
  asm volatile("fence" ::: "memory");
  // Input rows/columns repeat with these periods; every output is still
  // checked.
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 7; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k)
        sum += a[i * K + k] * b[k * N + j];
      expected[i][j] = sum;
    }
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      if (c[i * N + j] != expected[i % 9][j % 7])
        return i * N + j + 1;
  return 0;
}
