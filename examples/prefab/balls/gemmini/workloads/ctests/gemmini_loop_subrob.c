#include <bbhw/isa/isa.h>
#include <isa/gemmini.h>
#include <stdint.h>

#define DIM (THREAD_MEM_BANK_WIDTH / 8)
#define M (2 * DIM)
#define N DIM
#define K (2 * DIM)
static int8_t a[M * K] __attribute__((aligned(64)));
static int8_t b[K * N] __attribute__((aligned(64)));
static int32_t c[M * N] __attribute__((aligned(64)));
static int32_t expected[11][13];
_Static_assert(DIM == 16,
               "Gemmini loop tests require the configured 16x16 array");

int main(void) {
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = (i * 3 + k * 5) % 11 - 5;
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      b[k * N + j] = (k * 7 + j * 2) % 13 - 6;
  for (int i = 0; i < M * N; ++i)
    c[i] = 0x13579bdf;
  bb_gemmini_config(0, 0, 0, 0, 0);
  bb_gemmini_loop_ws_config_bounds(M / DIM, N / DIM, K / DIM);
  bb_gemmini_loop_ws_config_addr_a((uintptr_t)a);
  bb_gemmini_loop_ws_config_addr_b((uintptr_t)b);
  bb_gemmini_loop_ws_config_addr_d(0);
  bb_gemmini_loop_ws_config_addr_c((uintptr_t)c);
  bb_gemmini_loop_ws_config_strides_ab(K, N);
  bb_gemmini_loop_ws_config_strides_dc(0, N * 4);
  bb_gemmini_loop_ws(0, 1, 2, 1);
  bb_barrier();
  asm volatile("fence" ::: "memory");
  // Input rows/columns repeat with these periods; every output is still
  // checked.
  for (int i = 0; i < 11; ++i)
    for (int j = 0; j < 13; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k)
        sum += a[i * K + k] * b[k * N + j];
      expected[i][j] = sum;
    }
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      if (c[i * N + j] != expected[i % 11][j % 13])
        return i * N + j + 1;
  return 0;
}
