#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* C[18,36] = A[18,32] . B[36,32]^T over six output tiles of a 16x16 array
 * (whole, and short in rows, columns and both), one K block each: a tile's
 * only block waits for the tile before to be written. Codes in [-8, 9] and
 * block exponents within 2 of 1 make every term exact in FP32, so C is known
 * bit for bit. */
#define M 18
#define N 36
#define K 32
#define ROW (MXINT8_ROWS(K) * 16)

static uint8_t a[M * ROW] __attribute__((aligned(64)));
static uint8_t b[N * ROW] __attribute__((aligned(64)));
static float c[M * N] __attribute__((aligned(64)));

/* A code is a row part plus a position part of period 8: over the 32
 * positions of a block the position parts sum to 16 (A) and -16 (B), their
 * products to 8, so a block's exact sum of products is known in closed form. */
static int row_a(int m) { return (m * 7) % 15 - 7; }
static int row_b(int n) { return (n * 5) % 13 - 6; }
static int code_a(int m, int i) { return row_a(m) + i % 4 - 1; }
static int code_b(int n, int i) { return row_b(n) + (i % 8) / 2 - 2; }
static int exp_a(int m, int block) { return 127 + (m + block) % 2; }
static int exp_b(int n, int block) { return 127 - (n + 2 * block) % 3; }

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
  bb_mxint8mm(1, 2, 3, M, N, K, 0, N / 4, 0, 0, 0, 0);
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

  for (int i = 0; i < M * N; ++i)
    c[i] = -1.0f;
  bb_thread_run(run, 0);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float product = 0.0f;
      for (int block = 0; block < K / 32; ++block) {
        int p = 32 * row_a(m) * row_b(n) - 16 * row_a(m) + 16 * row_b(n) + 8;
        int shift = exp_a(m, block) + exp_b(n, block) - 254;
        product += (float)(p * power(shift));
      }
      float expected = product;
      float got = c[m * N + n];
      if (memcmp(&got, &expected, 4)) {
        printf("mm blocks: C[%d][%d] %f, not %f\n", m, n, (double)got,
               (double)expected);
        return 2;
      }
    }
  printf("mxint8mm blocks PASSED\n");
  return 0;
}
