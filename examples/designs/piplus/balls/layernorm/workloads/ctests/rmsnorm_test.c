#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/layernorm.h>
#include <math.h>
#include <stdio.h>
#include <thread_run.h>

/* RMSNorm (rms): three rows of 37 values (10 bank rows, 3 lanes of
 * padding): a spread of values, a row offset by 1000, and a row of tiny
 * values (eps dominates), gamma per column then eps; checked against the RMS
 * normalization in double precision. */
#define ROWS 3
#define WIDTH 37
#define LINES LAYERNORM_ROWS(WIDTH)
#define PITCH (LINES * 4)

static float input[ROWS * PITCH] __attribute__((aligned(64)));
static float params[(LINES + 1) * 4] __attribute__((aligned(64)));
static float output[ROWS * PITCH] __attribute__((aligned(64)));

static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)input, 1, ROWS * LINES, 1);
  bb_mvin((uintptr_t)params, 2, LINES + 1, 1);
  bb_layernorm(1, 2, 3, ROWS, WIDTH, 0, 0, 0, 1);
  bb_mvout((uintptr_t)output, 3, ROWS * LINES, 1);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}

int main(void) {
  for (int i = 0; i < WIDTH; ++i) {
    input[i] = (float)((i * 13) % 17 - 8) * 0.5f;
    input[PITCH + i] = 1000.0f + (float)(i % 7) * 0.25f;
    input[2 * PITCH + i] = 1e-4f * (float)(i % 5 - 2);
    params[i] = 0.5f + (float)i / 64.0f;
  }
  params[PITCH] = 1e-6f;
  for (int row = 0; row < ROWS; ++row)
    for (int i = WIDTH; i < PITCH; ++i)
      input[row * PITCH + i] = 12345.0f; // padding: ignored

  bb_thread_run(run, 0);

  for (int row = 0; row < ROWS; ++row) {
    const float *x = input + row * PITCH, *y = output + row * PITCH;
    double square = 0;
    for (int i = 0; i < WIDTH; ++i)
      square += (double)x[i] * x[i];
    square = square / WIDTH + 1e-6;
    double root;
    __asm__("fsqrt.d %0, %1" : "=f"(root) : "f"(square)); // ctests link no libm
    double r = 1 / root;
    for (int i = 0; i < WIDTH; ++i) {
      double expected = x[i] * r * params[i];
      if (fabs(y[i] - expected) > 1e-4 * (1 + fabs(expected))) {
        printf("rmsnorm: row %d column %d: %g != %g\n", row, i, (double)y[i],
               expected);
        return 1;
      }
    }
    for (int i = WIDTH; i < PITCH; ++i)
      if (y[i] != 0.0f) {
        printf("rmsnorm: row %d padding lane %d not zero\n", row, i);
        return 2;
      }
  }
  printf("rmsnorm PASSED\n");
  return 0;
}
