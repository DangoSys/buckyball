#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8quant.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* A row-major X[K][N] in a bank, K = 544 (17 blocks: a second exponent row),
 * N = 8, its rows PITCH = 3 bank rows apart (the third bank row of each
 * holds 1e30, which no block may read): the columns mode quantizes its
 * columns along K, all eight at once and columns 4..7 alone from in_base 1,
 * and must give the bytes the rows mode gives for the rows of X^T. */
#define ROWS 8
#define K 544
#define PITCH 3
#define STRIDE (MXINT8_ROWS(K) * 16)

static float matrix[K][PITCH * 4] __attribute__((aligned(64)));
static float transposed[ROWS][K] __attribute__((aligned(64)));
static uint8_t columns[ROWS * STRIDE] __attribute__((aligned(64)));
static uint8_t upper[ROWS / 2 * STRIDE] __attribute__((aligned(64)));
static uint8_t rows[ROWS * STRIDE] __attribute__((aligned(64)));

/* 2^n, -126 <= n <= 127. */
static float power(int n) {
  uint32_t bits = (uint32_t)(n + 127) << 23;
  float value;
  memcpy(&value, &bits, 4);
  return value;
}

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)matrix, 1, K * PITCH, 1);
  bb_mxint8quant(1, 2, ROWS, K, 0, 0, 1, PITCH, 0);
  bb_mvout((uintptr_t)columns, 2, ROWS * MXINT8_ROWS(K), 1);
  bb_mxint8quant(1, 2, ROWS / 2, K, 1, 0, 1, PITCH, 0);
  bb_mvout((uintptr_t)upper, 2, ROWS / 2 * MXINT8_ROWS(K), 1);
  bb_mvin((uintptr_t)transposed, 1, ROWS * K / 4, 1);
  bb_mxint8quant(1, 2, ROWS, K, 0, 0, 0, K / 4, 0);
  bb_mvout((uintptr_t)rows, 2, ROWS * MXINT8_ROWS(K), 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

int main(void) {
  uint32_t state = 7;
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < ROWS; ++n) {
      state = state * 1664525u + 1013904223u;
      float unit = (float)(state >> 8) / (float)(1u << 23) - 1.0f;
      float x = unit * power((k / 32 * 5 + n * 3) % 41 - 20);
      if (k % 32 == 0 && n == 5)
        x = 0.0f;
      /* An all-zero block (exponent byte 127). */
      if (n == 6 && k >= 64 && k < 96)
        x = 0.0f;
      matrix[k][n] = x;
      transposed[n][k] = x;
    }
    for (int n = ROWS; n < PITCH * 4; ++n)
      matrix[k][n] = 1e30f;
  }

  bb_thread_run(run, 0);

  for (int i = 0; i < ROWS * STRIDE; ++i)
    if (columns[i] != rows[i]) {
      printf("quant columns: row %d byte %d is %u, not %u\n", i / STRIDE,
             i % STRIDE, columns[i], rows[i]);
      return 1;
    }
  for (int i = 0; i < ROWS / 2 * STRIDE; ++i)
    if (upper[i] != rows[ROWS / 2 * STRIDE + i]) {
      printf("quant columns: from column 4, row %d byte %d is %u, not %u\n",
             i / STRIDE, i % STRIDE, upper[i], rows[ROWS / 2 * STRIDE + i]);
      return 2;
    }
  if (rows[6 * STRIDE + K + 2] != 127) {
    printf("quant columns: zero block exponent %u\n", rows[6 * STRIDE + K + 2]);
    return 3;
  }
  printf("mxint8quant columns PASSED\n");
  return 0;
}
