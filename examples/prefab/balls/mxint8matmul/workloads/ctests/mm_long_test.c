#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8matmul.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* C[6,12] += A[6,544] . B[12,544]^T onto an old C of integers, from bank
 * rows 3, 1 and 2: 17 blocks, two exponent rows a packed row. Codes in
 * [-15, 15] and block exponents within 2 of 1 keep every sum exact. */
#define M 6
#define N 12
#define K 544
#define ROW (MXINT8_ROWS(K) * 16)

static uint8_t a[3 * 16 + M * ROW] __attribute__((aligned(64)));
static uint8_t b[1 * 16 + N * ROW] __attribute__((aligned(64)));
static float c[2 * 4 + M * N] __attribute__((aligned(64)));

static int code_a(int m, int i) { return (m * 7 + i * 3) % 31 - 15; }
static int code_b(int n, int i) { return (n * 5 + i * 11) % 29 - 14; }
static int exp_a(int m, int block) { return 127 + (m + block) % 2; }
static int exp_b(int n, int block) { return 127 - (n + 2 * block) % 3; }
static float old(int m, int n) { return (float)(m * 13 - n * 7); }

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
  bb_mvin((uintptr_t)a, 1, 3 + M * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)b, 2, 1 + N * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)c, 3, 2 + M * N / 4, 1);
  bb_mxint8mm(1, 2, 3, M, N, K, 3, 1, 2, 1);
  bb_mvout((uintptr_t)c, 3, 2 + M * N / 4, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}

int main(void) {
  for (int m = 0; m < M; ++m)
    for (int i = 0; i < K; ++i) {
      a[48 + m * ROW + i] = (uint8_t)code_a(m, i);
      a[48 + m * ROW + K + i / 32] = (uint8_t)exp_a(m, i / 32);
    }
  for (int n = 0; n < N; ++n)
    for (int i = 0; i < K; ++i) {
      b[16 + n * ROW + i] = (uint8_t)code_b(n, i);
      b[16 + n * ROW + K + i / 32] = (uint8_t)exp_b(n, i / 32);
    }

  for (int i = 0; i < M * N; ++i)
    c[8 + i] = old(i / N, i % N);
  bb_thread_run(run, 0);

  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float expected = old(m, n);
      for (int block = 0; block < K / 32; ++block) {
        int p = 0;
        for (int i = block * 32; i < block * 32 + 32; ++i)
          p += code_a(m, i) * code_b(n, i);
        int shift = exp_a(m, block) + exp_b(n, block) - 254;
        expected += (float)(p * power(shift));
      }
      float got = c[8 + m * N + n];
      if (memcmp(&got, &expected, 4)) {
        printf("mm long: C[%d][%d] %f, not %f\n", m, n, (double)got,
               (double)expected);
        return 1;
      }
    }
  printf("mxint8matmul long PASSED\n");
  return 0;
}
