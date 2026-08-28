#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <isa/transpose.h>
#include <isa/vecmat16.h>
#include <stdio.h>

#define DIM 16

static elem_t input_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output[DIM * DIM] __attribute__((aligned(64)));
static result_t expected[DIM * DIM] __attribute__((aligned(64)));
static result_t zero[DIM * DIM] __attribute__((aligned(64))) = {0};

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_u8_random_matrix(input_a, DIM, DIM, 456);
  init_u8_random_matrix(input_b, DIM, DIM, 789);
  clear_u32_matrix(output, DIM, DIM);
  cpu_matmul(input_a, input_b, expected, DIM, DIM, DIM);

  const uint32_t op1 = 0;
  const uint32_t op2 = 1;
  const uint32_t acc = 2;
  const uint32_t a_t = 3;
  bb_mem_alloc(op1, 1, 1);
  bb_mem_alloc(op2, 1, 1);
  bb_mem_alloc(acc, 1, 4);
  bb_mem_alloc(a_t, 1, 1);

  bb_mvin((uintptr_t)zero, acc, DIM, 1);
  bb_mvin((uintptr_t)input_a, op1, DIM, 1);
  bb_mvin((uintptr_t)input_b, op2, DIM, 1);
  bb_transpose(op1, a_t, DIM, 8);
  bb_vecmat16(a_t, op2, acc, DIM, 0);
  bb_mvout((uintptr_t)output, acc, DIM, 1);
  bb_fence();

  int passed = compare_u32_matrices(output, expected, DIM, DIM);
  printf("vecunit_matmul_random1 %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
