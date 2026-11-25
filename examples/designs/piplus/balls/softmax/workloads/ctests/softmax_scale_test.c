#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/softmax.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* Two rows of 37 values (10 bank rows, 3 lanes of padding), attention-like
 * scores up to +-200 and a -inf, scaled by 2^-3 in the Ball: the same bits as
 * the softmax of the scores multiplied by 0.125 in C (exact) without a scale.
 */
#define ROWS 2
#define WIDTH 37
#define PITCH (SOFTMAX_ROWS(WIDTH) * 4)

static float input[ROWS * PITCH] __attribute__((aligned(64)));
static float scaled[ROWS * PITCH] __attribute__((aligned(64)));
static float output[ROWS * PITCH] __attribute__((aligned(64)));
static float reference[ROWS * PITCH] __attribute__((aligned(64)));

static void softmax(float *in, float *out, int scale_log2) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)in, 1, ROWS * PITCH / 4, 1);
  bb_softmax(1, 2, ROWS, WIDTH, 0, 0, scale_log2);
  bb_mvout((uintptr_t)out, 2, ROWS * PITCH / 4, 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

static void run(void *unused) {
  (void)unused;
  softmax(input, output, -3);
  softmax(scaled, reference, 0);
}

int main(void) {
  for (int row = 0; row < ROWS; ++row)
    for (int i = 0; i < PITCH; ++i)
      input[row * PITCH + i] =
          i < WIDTH ? (float)((i * 37 + row * 11) % 401 - 200) * 0.75f : 0.0f;
  input[5] = -__builtin_inff();
  for (int i = 0; i < ROWS * PITCH; ++i)
    scaled[i] = input[i] * 0.125f;

  bb_thread_run(run, 0);

  if (memcmp(output, reference, sizeof(output))) {
    for (int i = 0; i < ROWS * PITCH; ++i)
      if (output[i] != reference[i]) {
        printf("softmax scale: element %d %a, not %a\n", i, (double)output[i],
               (double)reference[i]);
        break;
      }
    return 1;
  }
  if (output[5] != 0.0f) {
    printf("softmax scale: -inf gives a nonzero probability\n");
    return 2;
  }
  printf("softmax scale PASSED\n");
  return 0;
}
