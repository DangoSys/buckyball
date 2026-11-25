#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/softmax.h>
#include <math.h>
#include <stdio.h>
#include <thread_run.h>

/* Three rows of 18 values, a row in 5 bank rows (20 lanes, 2 of padding):
 * -inf among repeated values, a spread of 260 (exponentials that vanish),
 * and a constant row. */
#define ROWS 3
#define WIDTH 18
#define PITCH (SOFTMAX_ROWS(WIDTH) * 4)

static float input[ROWS * PITCH] __attribute__((aligned(64)));
static float output[ROWS * PITCH] __attribute__((aligned(64)));

static int near(float a, float b, float tolerance) {
  return fabsf(a - b) <= tolerance * fabsf(b);
}

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 1, ROWS * PITCH / 4, 1);
  bb_softmax(1, 2, ROWS, WIDTH, 0, 0, 0);
  bb_mvout((uintptr_t)output, 2, ROWS * PITCH / 4, 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

int main(void) {
  for (int i = 0; i < WIDTH; ++i) {
    input[i] = (float)(i % 5 - 2);
    input[PITCH + i] = -160.0f + 15.0f * i;
    input[2 * PITCH + i] = 3.25f;
  }
  input[3] = input[7] = -INFINITY;
  for (int row = 0; row < ROWS; ++row)
    for (int i = WIDTH; i < PITCH; ++i)
      input[row * PITCH + i] = NAN; // padding: ignored
  for (int i = 0; i < ROWS * PITCH; ++i)
    output[i] = 1.0f;

  bb_thread_run(run, 0);

  for (int row = 0; row < ROWS; ++row) {
    const float *y = output + row * PITCH;
    float sum = 0.0f;
    for (int i = 0; i < WIDTH; ++i) {
      if (!(y[i] >= 0.0f && y[i] <= 1.0f)) {
        printf("softmax edges: row %d lane %d out of [0, 1]\n", row, i);
        return 1;
      }
      sum += y[i];
    }
    for (int i = WIDTH; i < PITCH; ++i)
      if (y[i] != 0.0f) {
        printf("softmax edges: row %d padding lane %d not zero\n", row, i);
        return 2;
      }
    if (!near(sum, 1.0f, 1e-5f)) {
      printf("softmax edges: row %d sums to %f\n", row, (double)sum);
      return 3;
    }
  }
  if (output[3] != 0.0f || output[7] != 0.0f) {
    printf("softmax edges: -inf gives a nonzero probability\n");
    return 4;
  }
  // x = 1 against x = 0 (lanes 8 and 2): a ratio of e.
  if (!near(output[8] / output[2], 2.7182817f, 1e-5f)) {
    printf("softmax edges: e^1 ratio %f\n", (double)(output[8] / output[2]));
    return 5;
  }
  for (int i = 0; i < WIDTH; ++i)
    if (output[2 * PITCH + i] != output[2 * PITCH]) {
      printf("softmax edges: constant row not uniform at %d\n", i);
      return 6;
    }
  if (output[PITCH] != 0.0f || output[PITCH + WIDTH - 1] < 0.9f) {
    printf("softmax edges: spread row %f %f\n", (double)output[PITCH],
           (double)output[PITCH + WIDTH - 1]);
    return 7;
  }
  printf("softmax edges PASSED\n");
  return 0;
}
