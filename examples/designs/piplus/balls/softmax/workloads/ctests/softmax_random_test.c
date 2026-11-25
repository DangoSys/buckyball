#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/softmax.h>
#include <math.h>
#include <stdio.h>
#include <thread_run.h>

/* Four rows of 291 values (attention over 241 cached and 50 new tokens), in
 * [-8, 8), from in_base 3 to out_base 5. */
#define ROWS 4
#define WIDTH 291
#define LINES SOFTMAX_ROWS(WIDTH)
#define PITCH (LINES * 4)
#define IN_BASE 3
#define OUT_BASE 5

static float input[IN_BASE * 4 + ROWS * PITCH] __attribute__((aligned(64)));
static float output[OUT_BASE * 4 + ROWS * PITCH] __attribute__((aligned(64)));

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 1, IN_BASE + ROWS * LINES, 1);
  bb_softmax(1, 2, ROWS, WIDTH, IN_BASE, OUT_BASE, 0);
  bb_mvout((uintptr_t)output, 2, OUT_BASE + ROWS * LINES, 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

int main(void) {
  uint32_t state = 12345;
  for (int row = 0; row < ROWS; ++row)
    for (int i = 0; i < WIDTH; ++i) {
      state = state * 1664525u + 1013904223u;
      input[IN_BASE * 4 + row * PITCH + i] =
          (float)(state >> 8) / (float)(1u << 24) * 16.0f - 8.0f;
    }

  bb_thread_run(run, 0);

  for (int row = 0; row < ROWS; ++row) {
    const float *x = input + IN_BASE * 4 + row * PITCH;
    const float *y = output + OUT_BASE * 4 + row * PITCH;
    float sum = 0.0f;
    int top = 0;
    for (int i = 0; i < WIDTH; ++i) {
      if (!(y[i] > 0.0f && y[i] < 1.0f)) {
        printf("softmax random: row %d lane %d out of (0, 1)\n", row, i);
        return 1;
      }
      sum += y[i];
      if (x[i] > x[top])
        top = i;
    }
    if (fabsf(sum - 1.0f) > 1e-5f) {
      printf("softmax random: row %d sums to %f\n", row, (double)sum);
      return 2;
    }
    // Larger values have larger probabilities; the largest has the largest.
    for (int i = 0; i < WIDTH; ++i) {
      if (y[i] > y[top]) {
        printf("softmax random: row %d lane %d above the maximum\n", row, i);
        return 3;
      }
      if (i > 0 && ((x[i] > x[i - 1] && y[i] < y[i - 1]) ||
                    (x[i] < x[i - 1] && y[i] > y[i - 1]))) {
        printf("softmax random: row %d lanes %d, %d out of order\n", row, i - 1,
               i);
        return 4;
      }
    }
    for (int i = WIDTH; i < PITCH; ++i)
      if (y[i] != 0.0f) {
        printf("softmax random: row %d padding lane %d not zero\n", row, i);
        return 5;
      }
  }
  printf("softmax random PASSED\n");
  return 0;
}
