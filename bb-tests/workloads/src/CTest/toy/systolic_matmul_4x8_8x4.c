#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define M_DIM 4
#define K_DIM 8
#define N_DIM 4
#define TILE_DIM 4
#define PACKED_INPUT_COLS (BANK_WIDTH / 8 / (int)sizeof(elem_t))
#define PACKED_OUTPUT_GROUPS 4
#define PACKED_OUTPUT_COLS                                                     \
  (PACKED_OUTPUT_GROUPS * (BANK_WIDTH / 8 / (int)sizeof(result_t)))

static elem_t input_matrix_a[M_DIM * K_DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[K_DIM * N_DIM] __attribute__((aligned(64)));
static result_t expected_matrix[M_DIM * N_DIM] __attribute__((aligned(64)));
static result_t output_matrix[M_DIM * N_DIM] __attribute__((aligned(64)));
static elem_t packed_a_tiles[2][TILE_DIM * PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static elem_t packed_b_tiles[2][TILE_DIM * PACKED_INPUT_COLS]
    __attribute__((aligned(64)));
static result_t packed_output_matrix[TILE_DIM * PACKED_OUTPUT_COLS]
    __attribute__((aligned(64)));

static void pack_a_tile(const elem_t *src, int k_base, elem_t *dst) {
  clear_u8_matrix(dst, TILE_DIM, PACKED_INPUT_COLS);
  for (int row = 0; row < TILE_DIM; ++row) {
    for (int col = 0; col < TILE_DIM; ++col) {
      dst[row * PACKED_INPUT_COLS + col] = src[row * K_DIM + k_base + col];
    }
  }
}

static void pack_b_tile(const elem_t *src, int k_base, elem_t *dst) {
  clear_u8_matrix(dst, TILE_DIM, PACKED_INPUT_COLS);
  for (int row = 0; row < TILE_DIM; ++row) {
    for (int col = 0; col < TILE_DIM; ++col) {
      dst[row * PACKED_INPUT_COLS + col] = src[(k_base + row) * N_DIM + col];
    }
  }
}

static void unpack_output_matrix(const result_t *src, result_t *dst) {
  for (int row = 0; row < TILE_DIM; ++row) {
    for (int col = 0; col < TILE_DIM; ++col) {
      dst[row * N_DIM + col] = src[row * PACKED_OUTPUT_COLS + col * 4];
    }
  }
}

static void issue_accumulated_matmul(int ws_mode) {
  uint32_t op1_bank_id = 0;
  uint32_t op2_bank_id = 1;
  uint32_t acc_bank_id = 2;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(op2_bank_id, 1, 1);
  bb_mem_alloc(acc_bank_id, 1, 4);

  pack_a_tile(input_matrix_a, 0, packed_a_tiles[0]);
  pack_a_tile(input_matrix_a, 4, packed_a_tiles[1]);
  pack_b_tile(input_matrix_b, 0, packed_b_tiles[0]);
  pack_b_tile(input_matrix_b, 4, packed_b_tiles[1]);

  bb_mvin((uintptr_t)packed_a_tiles[0], op1_bank_id, TILE_DIM, 1);
  bb_mvin((uintptr_t)packed_b_tiles[0], op2_bank_id, TILE_DIM, 1);
  if (ws_mode) {
    bb_BFP_WS_ACC_FIRST(op1_bank_id, op2_bank_id, acc_bank_id, TILE_DIM);
  } else {
    bb_BFP_OS_ACC_FIRST(op1_bank_id, op2_bank_id, acc_bank_id, TILE_DIM);
  }

  bb_mvin((uintptr_t)packed_a_tiles[1], op1_bank_id, TILE_DIM, 1);
  bb_mvin((uintptr_t)packed_b_tiles[1], op2_bank_id, TILE_DIM, 1);
  if (ws_mode) {
    bb_BFP_WS_ACC_LAST(op1_bank_id, op2_bank_id, acc_bank_id, TILE_DIM);
  } else {
    bb_BFP_OS_ACC_LAST(op1_bank_id, op2_bank_id, acc_bank_id, TILE_DIM);
  }

  bb_mvout((uintptr_t)packed_output_matrix, acc_bank_id, TILE_DIM, 1);
  bb_fence();
}

static int run_test(const char *name, int ws_mode) {
  clear_u32_matrix(output_matrix, M_DIM, N_DIM);
  clear_u32_matrix(expected_matrix, M_DIM, N_DIM);
  clear_u32_matrix(packed_output_matrix, TILE_DIM, PACKED_OUTPUT_COLS);

  cpu_matmul(input_matrix_a, input_matrix_b, expected_matrix, M_DIM, N_DIM,
             K_DIM);
  issue_accumulated_matmul(ws_mode);
  unpack_output_matrix(packed_output_matrix, output_matrix);

  if (!compare_u32_matrices(output_matrix, expected_matrix, M_DIM, N_DIM)) {
    printf("%s FAILED\n", name);
    return 0;
  }

  printf("%s PASSED\n", name);
  return 1;
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  init_u8_random_matrix(input_matrix_a, M_DIM, K_DIM, 123);
  init_u8_random_matrix(input_matrix_b, K_DIM, N_DIM, 456);

  int os_passed = run_test("systolic_matmul_4x8_8x4_os", 0);
  int ws_passed = run_test("systolic_matmul_4x8_8x4_ws", 1);
  int rc = (os_passed && ws_passed) ? 0 : 1;

#ifdef MULTICORE
  exit(rc);
#endif
  return rc;
}
