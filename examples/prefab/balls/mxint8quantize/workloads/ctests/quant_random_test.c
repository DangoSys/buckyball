#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8quantize.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <thread_run.h>

/* Three rows of K = 544 (17 blocks: a second exponent row), every block of
 * its own power-of-two scale. Checked against the definition: the block
 * exponent from the largest magnitude, each code within half a step of its
 * value, the clamp at 127, the zero padding. */
#define ROWS 3
#define K 544
#define STRIDE (MXINT8_ROWS(K) * 16)

static float input[ROWS * K] __attribute__((aligned(64)));
static uint8_t output[ROWS * STRIDE] __attribute__((aligned(64)));

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
  bb_mvin((uintptr_t)input, 1, ROWS * K / 4, 1);
  bb_mxint8quant(1, 2, ROWS, K, 0, 0);
  bb_mvout((uintptr_t)output, 2, ROWS * MXINT8_ROWS(K), 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

int main(void) {
  uint32_t state = 2024;
  for (int i = 0; i < ROWS * K; ++i) {
    state = state * 1664525u + 1013904223u;
    float unit = (float)(state >> 8) / (float)(1u << 23) - 1.0f;
    input[i] = unit * power((int)((i / 32) * 7 % 61) - 30);
  }

  bb_thread_run(run, 0);

  for (int row = 0; row < ROWS; ++row) {
    const uint8_t *packed = output + row * STRIDE;
    for (int block = 0; block < K / 32; ++block) {
      const float *x = input + row * K + block * 32;
      uint32_t largest = 0;
      for (int i = 0; i < 32; ++i) {
        uint32_t bits;
        memcpy(&bits, &x[i], 4);
        if ((bits & 0x7fffffff) > largest)
          largest = bits & 0x7fffffff;
      }
      int biased = (int)(largest >> 23);
      int e = largest == 0 ? 0 : biased == 0 ? -127 : biased - 133;
      if (e < -127)
        e = -127;
      if (packed[K + block] != (uint8_t)(e + 127)) {
        printf("quant random: row %d block %d exponent %u, not %d\n", row,
               block, packed[K + block], e + 127);
        return 1;
      }
      for (int i = 0; i < 32; ++i) {
        float scaled = x[i] * power(-e);
        int code = (int8_t)packed[block * 32 + i];
        float target = scaled > 127.0f    ? 127.0f
                       : scaled < -127.0f ? -127.0f
                                          : scaled;
        if (fabsf((float)code - target) > 0.5f) {
          printf("quant random: row %d value %d code %d for %f\n", row,
                 block * 32 + i, code, (double)scaled);
          return 2;
        }
      }
    }
    for (int i = K + K / 32; i < STRIDE; ++i)
      if (packed[i]) {
        printf("quant random: row %d padding byte %d\n", row, i);
        return 3;
      }
  }
  printf("mxint8quantize random PASSED\n");
  return 0;
}
