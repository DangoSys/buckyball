#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <isa/mxint8quant.h>
#include <stdio.h>
#include <thread_run.h>

/* Two rows of K = 64, two blocks each, with hand-picked block maxima:
 * row 0 block 0 maximum 127.5 (exponent 0): ties to even and the clamp at
 * 127, signed zeros; row 0 block 1 all zero (byte 127); row 1 block 0
 * maximum 254 (exponent 1); row 1 block 1 maximum 1 (exponent -6). The rows
 * are PITCH bank rows apart; the two bank rows between them hold 1e30, which
 * no block may read. */
#define ROWS 2
#define K 64
#define STRIDE (MXINT8_ROWS(K) * 16)
#define IN_BASE 3
#define OUT_BASE 2
#define PITCH (K / 4 + 2)

static float input[(IN_BASE + PITCH + K / 4) * 4] __attribute__((aligned(64)));
static uint8_t output[OUT_BASE * 16 + ROWS * STRIDE]
    __attribute__((aligned(64)));

static const float values[ROWS][8] = {
    {127.5f, -127.5f, 0.5f, 1.5f, 2.5f, -2.5f, 3.5f, -0.5f},
    {254.0f, 3.0f, 5.0f, -7.0f, 1.0f, 0.5f, 0.25f, 1e-3f}};
static const int8_t codes[ROWS][8] = {{127, -127, 0, 2, 2, -2, 4, 0},
                                      {127, 2, 2, -4, 64, 32, 16, 0}};
static const uint8_t exponents[ROWS][2] = {{127, 127}, {128, 121}};

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input, 1, IN_BASE + PITCH + K / 4, 1);
  bb_mxint8quant(1, 2, ROWS, K, IN_BASE, OUT_BASE, 0, PITCH, 0);
  bb_mvout((uintptr_t)output, 2, OUT_BASE + ROWS * MXINT8_ROWS(K), 1);
  bb_mem_release(1);
  bb_mem_release(2);
}

int main(void) {
  float *x = input + IN_BASE * 4;
  for (int i = 0; i < 6; ++i) {
    x[i] = values[0][i];                                     // row 0 block 0
    x[PITCH * 4 + i] = values[1][i % 4];                     // row 1 block 0
    x[PITCH * 4 + 32 + i] = i < 4 ? values[1][4 + i] : 0.0f; // row 1 block 1
  }
  for (int i = K; i < PITCH * 4; ++i)
    x[i] = 1e30f;
  x[6] = values[0][6];
  x[7] = values[0][7];
  x[8] = -0.0f;
  x[9] = 126.49999f; // 126

  bb_thread_run(run, 0);

  for (int row = 0; row < ROWS; ++row) {
    const uint8_t *packed = output + OUT_BASE * 16 + row * STRIDE;
    for (int block = 0; block < 2; ++block)
      if (packed[K + block] != exponents[row][block]) {
        printf("quant edges: row %d block %d exponent %u\n", row, block,
               packed[K + block]);
        return 1;
      }
    for (int i = 0; i < K; ++i) {
      int expected = 0;
      if (row == 0 && i < 8)
        expected = codes[0][i];
      if (row == 0 && i == 9)
        expected = 126;
      if (row == 1 && i < 6)
        expected = codes[1][i % 4];
      if (row == 1 && i >= 32 && i < 36)
        expected = codes[1][4 + i - 32];
      if ((int8_t)packed[i] != expected) {
        printf("quant edges: row %d code %d is %d, not %d\n", row, i,
               (int8_t)packed[i], expected);
        return 2;
      }
    }
    for (int i = K + 2; i < STRIDE; ++i)
      if (packed[i]) {
        printf("quant edges: row %d padding byte %d\n", row, i);
        return 3;
      }
  }
  printf("mxint8quant edges PASSED\n");
  return 0;
}
