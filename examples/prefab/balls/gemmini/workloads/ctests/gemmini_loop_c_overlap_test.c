#include <bbhw/isa/isa.h>
#include <isa/gemmini.h>
#include <stdint.h>

// LOOP_WS with C written over the second k-tile of B: C = A B for
// M = N = 16, K = 32. The loop must read B(1, 0) before it stores C, which it
// does only once, after the last k (LoopMatmulUnroller). A partial C stored
// after k = 0 would overwrite B(1, 0) before it is read.
#define DIM (THREAD_MEM_BANK_WIDTH / 8)
#define M DIM
#define N DIM
#define K (2 * DIM)
_Static_assert(DIM == 16,
               "Gemmini loop tests require the configured 16x16 array");
static int8_t a[M * K] __attribute__((aligned(64)));
// B (K x N bytes) at offset 0, C (M x N words) from B's second k-tile on.
static int8_t memory[DIM * N + M * N * 4] __attribute__((aligned(64)));
static int8_t b_copy[K * N];
static int32_t expected[M][N];

int main(void) {
  int8_t *b = memory;
  int32_t *c = (int32_t *)(memory + DIM * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = (i * 3 + k * 5) % 11 - 5;
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) {
      b[k * N + j] = (k * 7 + j * 2) % 13 - 6;
      b_copy[k * N + j] = b[k * N + j];
    }
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k)
        sum += a[i * K + k] * b_copy[k * N + j];
      expected[i][j] = sum;
    }
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
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      if (c[i * N + j] != expected[i][j])
        return i * N + j + 1;
  return 0;
}
