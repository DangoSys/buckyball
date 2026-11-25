#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8mm.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* C[5,12] += A[5,96] . B[12,96]^T onto an old C of integers, in place
 * (update): the two bank rows behind C stay as they were. Codes in
 * [-15, 15] and block exponents within 2 of 1 make every block term and
 * every sum exact in FP32, so C is known bit for bit. */
#define M 5
#define N 12
#define K 96
#define ROW (MXINT8_ROWS(K) * 16)

static uint8_t a[M * ROW] __attribute__((aligned(64)));
static uint8_t b[N * ROW] __attribute__((aligned(64)));
static float c[M * N + 8] __attribute__((aligned(64)));

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
  bb_mvin((uintptr_t)a, 1, M * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)b, 2, N * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)c, 3, M * N / 4 + 2, 1);
  bb_mxint8mm(1, 2, 3, M, N, K, 0, N / 4, 0, 1, 0, 1);
  bb_mvout((uintptr_t)c, 3, M * N / 4 + 2, 1);
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
    c[i] = old(i / N, i % N);
  for (int i = M * N; i < M * N + 8; ++i)
    c[i] = 77.0f;
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
      float got = c[m * N + n];
      if (memcmp(&got, &expected, 4)) {
        printf("mm accumulate update: C[%d][%d] %f, not %f\n", m, n,
               (double)got, (double)expected);
        return 1;
      }
    }
  for (int i = M * N; i < M * N + 8; ++i)
    if (c[i] != 77.0f) {
      printf("mm accumulate update: wrote behind C\n");
      return 1;
    }
  printf("mxint8mm accumulate update PASSED\n");
  return 0;
}
