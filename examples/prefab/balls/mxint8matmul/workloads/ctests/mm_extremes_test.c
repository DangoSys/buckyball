#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8matmul.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* C[3,4] = A[3,64] . B[4,64]^T at the edges of FP32: terms that vanish to
 * signed zero (shift -251), subnormal results (shifts -144, -143), blocks
 * that cancel to +0 (A row 1 block 1 = -block 0 against B row 2 of equal
 * blocks) and results near 2^113 (shift 100). */
#define M 3
#define N 4
#define K 64
#define ROW (MXINT8_ROWS(K) * 16)

static uint8_t a[M * ROW] __attribute__((aligned(64)));
static uint8_t b[N * ROW] __attribute__((aligned(64)));
static float c[M * N] __attribute__((aligned(64)));

static const int exps_a[M][2] = {{2, 2}, {127, 127}, {177, 177}};
static const int exps_b[N][2] = {{1, 1}, {108, 109}, {127, 127}, {177, 177}};
static int code_a(int m, int i) {
  int code = (m * 7 + i % 32 * 3) % 31 - 15;
  return m == 1 && i >= 32 ? -code : code;
}
static int code_b(int n, int i) {
  return (n * 5 + (n == 2 ? i % 32 : i) * 11) % 29 - 14;
}
static int exp_a(int m, int block) { return exps_a[m][block]; }
static int exp_b(int n, int block) { return exps_b[n][block]; }

/* 2^n as a double, -1022 <= n <= 1023. */
static double power(int n) {
  uint64_t bits = (uint64_t)(n + 1023) << 52;
  double value;
  memcpy(&value, &bits, 8);
  return value;
}

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)a, 1, M * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)b, 2, N * MXINT8_ROWS(K), 1);
  bb_mxint8mm(1, 2, 3, M, N, K, 0, 0, 0, 0);
  bb_mvout((uintptr_t)c, 3, M * N / 4, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}

int main(void) {
  for (int m = 0; m < M; ++m)
    for (int i = 0; i < K; ++i) {
      a[m * ROW + i] = (uint8_t)code_a(m, i);
      a[m * ROW + K + i / 32] = (uint8_t)exp_a(m, i / 32);
    }
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < K; ++i) {
      b[n * ROW + i] = (uint8_t)code_b(n, i);
      b[n * ROW + K + i / 32] = (uint8_t)exp_b(n, i / 32);
    }

  bb_thread_run(run, 0);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (int block = 0; block < K / 32; ++block) {
        int p = 0;
        for (int i = block * 32; i < block * 32 + 32; ++i)
          p += code_a(m, i) * code_b(n, i);
        int shift = exp_a(m, block) + exp_b(n, block) - 254;
        expected += (float)(p * power(shift));
      }
      float got = c[m * N + n];
      if (memcmp(&got, &expected, 4)) {
        printf("mm extremes: C[%d][%d] %f, not %f\n", m, n, (double)got,
               (double)expected);
        return 1;
      }
    }
  printf("mxint8matmul extremes PASSED\n");
  return 0;
}
