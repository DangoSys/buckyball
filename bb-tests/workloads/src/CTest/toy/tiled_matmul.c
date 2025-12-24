#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

#define DIM_I 32
#define DIM_J 32
#define DIM_K 32

static elem_t input_a[DIM_I * DIM_J] __attribute__((aligned(16)));
static elem_t input_b[DIM_J * DIM_K] __attribute__((aligned(16)));
static result_t output_c[DIM_I * DIM_K] __attribute__((aligned(64)));
static result_t expected_c[DIM_I * DIM_K] __attribute__((aligned(64)));
static result_t c_zero[DIM * DIM] __attribute__((aligned(64)));

void tiled_matmul_connect_mode(uint32_t dim_i, uint32_t dim_j, uint32_t dim_k,
                               elem_t *a, elem_t *b, result_t *c) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  // acc0: write to accumulator, offset 0
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  uint32_t i_stride = dim_i / DIM;
  uint32_t j_stride = dim_j / DIM;
  uint32_t k_stride = dim_k / DIM;
  uint64_t en = 1;

  unsigned long long start_compute, end_compute;
  start_compute = read_cycle();
  bb_bbus_config(3, 0, en);
  // mvin matrix a
  for (int i = 0; i < i_stride; i++) {
    for (int j = 0; j < j_stride; j++) {
      bb_mvin((uintptr_t)a + i * dim_j * DIM + j * DIM,
              op1_bank_id + dim_j * k_stride + dim_j * i + j * DIM, DIM,
              j_stride);
    }
  }
  // mvin matrix b
  for (int k = 0; k < k_stride; k++) {
    bb_mvin((uintptr_t)b + k * DIM, op2_bank_id + dim_j * k, dim_j, k_stride);
  }
  bb_fence();
  // perform matmul
  for (int i = 0; i < i_stride; i++) {
    for (int k = 0; k < k_stride; k++) {
      bb_mul_warp16(op1_bank_id + dim_j * i, op2_bank_id + dim_j * k,
                    acc_bank_id + i * dim_k / 2 + k * DIM / 2, dim_j, 1);
      bb_transpose((uintptr_t)op1_bank_id + dim_j * k_stride + dim_j * i,
                   op1_bank_id + dim_j * i, dim_j, 1);
    }
  }
  bb_fence();
  // mvout matrix c
  for (int i = 0; i < i_stride; i++) {
    for (int k = 0; k < k_stride; k++) {
      bb_mvout((uintptr_t)c + i * dim_k * DIM * 4 + k * DIM * 4,
               acc_bank_id + i * dim_k / 2 + k * DIM / 2, DIM << 2, k_stride);
      bb_fence();
    }
  }
  bb_bbus_config(3, 0, 0);
  end_compute = read_cycle();
  printf("Cycles for matmul: %d\n", end_compute - start_compute);
}

void tiled_matmul_normal_mode(uint32_t dim_i, uint32_t dim_j, uint32_t dim_k,
                              elem_t *a, elem_t *b, result_t *c) {
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  // acc0: write to accumulator, offset 0
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  uint32_t i_stride = dim_i / DIM;
  uint32_t j_stride = dim_j / DIM;
  uint32_t k_stride = dim_k / DIM;

  unsigned long long start_compute, end_compute;
  start_compute = read_cycle();

  // mvin matrix a
  for (int i = 0; i < i_stride; i++) {
    for (int j = 0; j < j_stride; j++) {
      bb_mvin((uintptr_t)a + i * dim_j * DIM + j * DIM,
              op1_bank_id + dim_j * k_stride + dim_j * i + j * DIM, DIM,
              j_stride);
    }
  }
  // mvin matrix b
  for (int k = 0; k < k_stride; k++) {
    bb_mvin((uintptr_t)b + k * DIM, op2_bank_id + dim_j * k, dim_j, k_stride);
  }
  bb_fence();

  // transpose matrix a
  for (int i = 0; i < i_stride; i++) {
    bb_transpose((uintptr_t)op2_bank_id + dim_j * k_stride + dim_j * i,
                 op1_bank_id + dim_j * i, dim_j, 0);
  }
  bb_fence();

  // perform matmul
  for (int i = 0; i < i_stride; i++) {
    for (int k = 0; k < k_stride; k++) {
      bb_mul_warp16(op1_bank_id + dim_j * i, op2_bank_id + dim_j * k,
                    acc_bank_id + i * dim_k / 2 + k * DIM / 2, dim_j, 0);
      bb_fence();
    }
  }
  bb_fence();
  // mvout matrix c
  for (int i = 0; i < i_stride; i++) {
    for (int k = 0; k < k_stride; k++) {
      bb_mvout((uintptr_t)c + i * dim_k * DIM * 4 + k * DIM * 4,
               acc_bank_id + i * dim_k / 2 + k * DIM / 2, DIM << 2, k_stride);
      bb_fence();
    }
  }

  end_compute = read_cycle();
  printf("Cycles for matmul: %d\n", end_compute - start_compute);
}

int run_test(const char *test_name) {

  tiled_matmul_normal_mode(DIM_I, DIM_J, DIM_K, input_a, input_b, output_c);
  cpu_matmul(input_a, input_b, expected_c, DIM_I, DIM_K, DIM_J);
  if (compare_u32_matrices(output_c, expected_c, DIM_I, DIM_K)) {
    printf("Test Connect Mode %s PASSED\n", test_name);
  } else {
    printf("Test Connect Mode %s FAILED\n", test_name);
    return 0;
  }

  clear_u32_matrix(output_c, DIM_I, DIM_K);
  // acc0: write to accumulator, offset 0
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);
  // TODO: ACC overwrite write can skip this step
  bb_mvin((uintptr_t)output_c, acc_bank_id, DIM_I * DIM_K * 4 / DIM, 1);
  tiled_matmul_normal_mode(DIM_I, DIM_J, DIM_K, input_a, input_b, output_c);
  if (compare_u32_matrices(output_c, expected_c, DIM_I, DIM_K)) {
    printf("Test Normal Mode %s PASSED\n", test_name);
    return 1;
  } else {
    printf("Test Normal Mode %s FAILED\n", test_name);
    return 0;
  }
}

int test_tiled_matmul() {
  /**
  init_u8_random_matrix(input_a, DIM_I, DIM_J, 111);
  init_u8_random_matrix(input_b, DIM_J, DIM_K, 222);
  */
  init_sequence_matrix(input_a, DIM_I, DIM_J);
  init_sequence_matrix(input_b, DIM_I, DIM_J);
  return run_test("Tiled Matmul");
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  // printf("%p ,%p", input_a, input_b);
  // printf("Testing Tiled Matmul\n");
  int passed = test_tiled_matmul();
  if (passed) {
    printf("Tiled Matmul test PASSED\n");
    return 0;
  } else {
    printf("Tiled Matmul test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
