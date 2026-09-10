#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/nli.h>
#include <stdint.h>
#include <stdio.h>

// Offline-optimised SiLU (Swish) NLI table, see
// examples/balls/nli/scripts/gen_silu_table.py. 16 non-uniform segments over
// the signed INT8 domain: 15 interior cutpoints, Q7 slopes, INT8 intercepts.
static const int8_t cutpoints[15] = {-96, -64, -44, -30, -20, -13, -8, -4,
                                     -1,  2,   5,   9,   16,  28,  48};
static const int8_t slopes[16] = {0,  0,  0,   0,   0,   0,   0,   -2,
                                  -8, 87, 127, 127, 127, 127, 127, 127};
static const int8_t intercepts[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0};

// 64 inputs covering the signed INT8 domain, incl. extreme outliers.
static const int8_t nli_input[64] = {
    -128, -127, -100, -96, -80,  -64, -48, -44, -32, -30, -24, -20, -16,
    -13,  -12,  -10,  -9,  -8,   -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,
    1,    2,    3,    4,   5,    6,   7,   8,   9,   10,  12,  13,  16,
    20,   24,   28,   32,  44,   48,  64,  80,  96,  100, 120, 127, -33,
    4,    41,   78,   115, -104, -67, -30, 7,   44,  81,  118, -101};

// round(SiLU(x)) reference, generated offline (avoids libm on baremetal).
static const int8_t silu_golden[64] = {
    0,   0,   0,   0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,   0,
    0,   0,   0,   0,  0,  0,  0,  0,   0,  0,  1,  2,  3,  4,  5,   6,
    7,   8,   9,   10, 12, 13, 16, 20,  24, 28, 32, 44, 48, 64, 80,  96,
    100, 120, 127, 0,  4,  41, 78, 115, 0,  0,  0,  7,  44, 81, 118, 0};

static int8_t input_buf[64] __attribute__((aligned(64)));
static int8_t table[48] __attribute__((aligned(64)));
static int8_t output[64] __attribute__((aligned(64)));

// Reference implementation of the NLI datapath (segment select + Q7 MAC),
// kept independent of the Ball so the test checks the hardware, not itself.
static int8_t nli_ref(int8_t x) {
  int seg = 0;
  for (int j = 0; j < 15; ++j)
    if (x >= cutpoints[j])
      ++seg;
  int prod = (int)slopes[seg] * (int)x;
  int acc = (prod >> 7) + intercepts[seg];
  if (acc > 127)
    acc = 127;
  if (acc < -128)
    acc = -128;
  return (int8_t)acc;
}

int main(void) {
  // Pack the coefficient bank: row 0 = 15 cutpoints (+pad), row 1 = 16
  // slopes, row 2 = 16 intercepts.
  for (int i = 0; i < 15; ++i)
    table[i] = cutpoints[i];
  table[15] = 0;
  for (int i = 0; i < 16; ++i)
    table[16 + i] = slopes[i];
  for (int i = 0; i < 16; ++i)
    table[32 + i] = intercepts[i];
  for (int i = 0; i < 64; ++i)
    input_buf[i] = nli_input[i];

  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)input_buf, 0, 4, 1);
  bb_mvin((uintptr_t)table, 1, 3, 1);
  bb_nli(0, 1, 2, 4);
  bb_mvout((uintptr_t)output, 2, 4, 1);
  bb_fence();

  // (1) Datapath check: exact match against the independent reference model.
  for (int i = 0; i < 64; ++i) {
    int8_t expected = nli_ref(input_buf[i]);
    if (output[i] != expected) {
      printf("nli FAIL datapath i=%d in=%d got=%d want=%d\n", i, input_buf[i],
             output[i], expected);
      return 1;
    }
  }

  // (2) Accuracy check: NLI approximation must match true SiLU within +/-1 LSB.
  for (int i = 0; i < 64; ++i) {
    int d = output[i] - silu_golden[i];
    if (d < -1 || d > 1) {
      printf("nli FAIL accuracy i=%d in=%d got=%d silu=%d\n", i, input_buf[i],
             output[i], silu_golden[i]);
      return 1;
    }
  }

  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  printf("nli PASS\n");
  return 0;
}
