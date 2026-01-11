#include "../toy/buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM (BANK_WIDTH / sizeof(elem_t) / 8)

// Test matrices
static int32_t input_matrix[DIM * DIM] __attribute__((aligned(16)));
static int32_t output_matrix[DIM * DIM] __attribute__((aligned(16)));

int mvin_mvout_acc_test() {
  for (int i = 0; i < 4; i++) {
    init_i32_random_matrix(input_matrix, DIM, DIM, 111);
    int acc_bank_id = bb_mem_alloc(1, 4); // row, col
    bb_mvin((uintptr_t)input_matrix, acc_bank_id, DIM, 1);
    clear_i32_matrix(output_matrix, DIM, DIM);
    bb_fence();
    bb_mvout((uintptr_t)output_matrix, acc_bank_id, DIM, 1);
    bb_fence();
    bb_mem_release(acc_bank_id);
    if (!compare_i32_matrices(output_matrix, input_matrix, DIM, DIM)) {
      printf("Test mvin/mvout simple %d FAILED\n", i);
      return 0;
    } else {
      printf("Test mvin/mvout simple %d PASSED\n", i);
    }
  }
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = mvin_mvout_acc_test();
  if (passed) {
    printf("mvin/mvout simple test PASSED\n");
  } else {
    printf("mvin/mvout simple test FAILED\n");
  }
#ifdef MULTICORE
  exit(0);
#endif
}
