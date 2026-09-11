#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/smatmul.h>
#include <stdint.h>
#include <stdio.h>

enum {
  DIM = 16,
  OUTPUT_ROWS = DIM * DIM * 32 / BANK_WIDTH,
  BIAS_ROWS = DIM * 32 / BANK_WIDTH,
  ITERATIONS = 24
};
static int8_t input[DIM * DIM] __attribute__((aligned(64)));
static int8_t weight[DIM * DIM] __attribute__((aligned(64)));
static int32_t bias[DIM] __attribute__((aligned(64)));
static int32_t zero[DIM * DIM] __attribute__((aligned(64)));
static int32_t computed[DIM * DIM] __attribute__((aligned(64)));
static int32_t cleared[DIM * DIM] __attribute__((aligned(64)));

int main(void) {
  for (int row = 0; row < DIM; ++row)
    for (int col = 0; col < DIM; ++col) {
      input[row * DIM + col] = (row * 3 + col * 5) % 15 - 7;
      weight[row * DIM + col] = (2 * row + 3 * col) % 5 - 2;
    }
  for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
    for (int bank = 0; bank < 4; ++bank)
      bb_mem_alloc(bank, 1, 1);
    bb_mvin((uintptr_t)input, 0, DIM, 1);
    bb_mvin((uintptr_t)weight, 1, DIM, 1);
    bb_mvin((uintptr_t)bias, 2, BIAS_ROWS, 1);
    bb_smatmul_bias(2, 0);
    bb_smatmul_os(0, 1, 3, DIM, DIM, DIM, 1, 1, 0);
    bb_mvout((uintptr_t)computed, 3, OUTPUT_ROWS, 1);
    bb_fence();
    bb_mem_release(3);
    bb_mem_alloc(3, 1, 1);
    bb_mvin((uintptr_t)zero, 3, OUTPUT_ROWS, 1);
    bb_mvout((uintptr_t)cleared, 3, OUTPUT_ROWS, 1);
    bb_fence();
    for (int row = 0; row < DIM; ++row)
      for (int col = 0; col < DIM; ++col) {
        int expected = 0;
        for (int k = 0; k < DIM; ++k)
          expected += input[row * DIM + k] * weight[k * DIM + col];
        int i = row * DIM + col;
        if (computed[i] != expected || cleared[i] != 0) {
          printf("smatmul mvout bank reuse mismatch iteration=%d index=%d\n",
                 iteration, i);
          return 1;
        }
      }
    for (int bank = 0; bank < 4; ++bank)
      bb_mem_release(bank);
  }
  printf("smatmul mvout bank reuse PASS\n");
  return 0;
}
