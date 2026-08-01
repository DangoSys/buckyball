#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_H 6
#define INPUT_W 6
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 1
#define PADDING 0

#define OUTPUT_H 4
#define OUTPUT_W 4
#define NUM_WINDOWS 16
#define KERNEL_ELEMS 9

#define INT8_LANES 16
#define FP32_LANES 4
#define ACC_LANES 16

#define QUANT_ROWS ((INPUT_H * INPUT_W + INT8_LANES - 1) / INT8_LANES)
#define PADDED_INPUT_FP32_ELEMS (QUANT_ROWS * INT8_LANES)
#define PADDED_QUANT_ELEMS (QUANT_ROWS * INT8_LANES)
#define IM2COL_ROWS NUM_WINDOWS
#define PADDED_IM2COL_ELEMS (IM2COL_ROWS * INT8_LANES)
#define TRANSPOSE_SOURCE_ELEMS (INT8_LANES * KERNEL_ELEMS)
#define PADDED_TRANSPOSE_ELEMS (KERNEL_ELEMS * INT8_LANES)
#define PADDED_GEMM_ELEMS (NUM_WINDOWS * ACC_LANES)

#define SCALE_FP32_TO_INT8 0x40000000U /* 2.0f */
#define SCALE_INT8_TO_FP32 0x3E800000U /* 0.25f */

static const float input_fp32[INPUT_H * INPUT_W] = {
    0.5f,  1.5f,  2.5f,  3.5f,  4.5f,  5.5f,  1.5f,  0.5f,  -0.5f,
    -1.5f, -2.5f, -3.5f, 2.5f,  -0.5f, -2.5f, 1.5f,  -1.5f, 3.5f,
    3.5f,  -1.5f, 1.5f,  -2.5f, 0.5f,  -1.5f, 4.5f,  -2.5f, -1.5f,
    0.5f,  2.5f,  -0.5f, 5.5f,  -3.5f, 3.5f,  -1.5f, -0.5f, 1.5f,
};

static const int8_t expected_quant_input[INPUT_H * INPUT_W] = {
    1, 3,  5, 7,  9, 11, 3, 1,  -1, -3, -5, -7, 5,  -1, -5, 3,  -3, 7,
    7, -3, 3, -5, 1, -3, 9, -5, -3, 1,  5,  -1, 11, -7, 7,  -3, -1, 3,
};

static const int8_t expected_im2col_a[NUM_WINDOWS * KERNEL_ELEMS] = {
    1,  3,  5,  3,  1,  -1, 5,  -1, -5, 3,  5,  7,  1,  -1, -3, -1, -5, 3,
    5,  7,  9,  -1, -3, -5, -5, 3,  -3, 7,  9,  11, -3, -5, -7, 3,  -3, 7,

    3,  1,  -1, 5,  -1, -5, 7,  -3, 3,  1,  -1, -3, -1, -5, 3,  -3, 3,  -5,
    -1, -3, -5, -5, 3,  -3, 3,  -5, 1,  -3, -5, -7, 3,  -3, 7,  -5, 1,  -3,

    5,  -1, -5, 7,  -3, 3,  9,  -5, -3, -1, -5, 3,  -3, 3,  -5, -5, -3, 1,
    -5, 3,  -3, 3,  -5, 1,  -3, 1,  5,  3,  -3, 7,  -5, 1,  -3, 1,  5,  -1,

    7,  -3, 3,  9,  -5, -3, 11, -7, 7,  -3, 3,  -5, -5, -3, 1,  -7, 7,  -3,
    3,  -5, 1,  -3, 1,  5,  7,  -3, -1, -5, 1,  -3, 1,  5,  -1, -3, -1, 3,
};

static const int8_t kernel_flat_1x9[KERNEL_ELEMS] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

static const int8_t expected_transpose_b[KERNEL_ELEMS] = {
    1, 0, -1, 2, 0, -2, 1, 0, -1,
};

static const int32_t expected_gemm_c[NUM_WINDOWS] = {
    14, 0, 2, 0, 28, -2, 2, -6, 30, -6, -6, -6, 32, -14, -6, -4,
};

static const int8_t expected_requant_output[NUM_WINDOWS] = {
    7, 0, 1, 0, 14, -1, 1, -3, 15, -3, -3, -3, 16, -7, -3, -2,
};

static const float expected_final_fp32[NUM_WINDOWS] = {
    1.75f, 0.00f,  0.25f,  0.00f,  3.50f, -0.25f, 0.25f,  -0.75f,
    3.75f, -0.75f, -0.75f, -0.75f, 4.00f, -1.75f, -0.75f, -0.50f,
};

static float packed_input_fp32[PADDED_INPUT_FP32_ELEMS]
    __attribute__((aligned(64)));
static int8_t actual_quant[PADDED_QUANT_ELEMS] __attribute__((aligned(64)));
static int8_t actual_im2col[PADDED_IM2COL_ELEMS] __attribute__((aligned(64)));
static int8_t transpose_source[TRANSPOSE_SOURCE_ELEMS]
    __attribute__((aligned(64)));
static int8_t actual_transpose[PADDED_TRANSPOSE_ELEMS]
    __attribute__((aligned(64)));
static int32_t zero_gemm[PADDED_GEMM_ELEMS] __attribute__((aligned(64)));
static int32_t actual_gemm[PADDED_GEMM_ELEMS] __attribute__((aligned(64)));
static int8_t requant_substitute[INT8_LANES] __attribute__((aligned(64)));
static float actual_final[NUM_WINDOWS] __attribute__((aligned(64)));

static void initialize_buffers(void) {
  for (int i = 0; i < PADDED_INPUT_FP32_ELEMS; ++i) {
    packed_input_fp32[i] = 0.0f;
  }
  for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
    packed_input_fp32[i] = input_fp32[i];
  }

  for (int i = 0; i < PADDED_QUANT_ELEMS; ++i) {
    actual_quant[i] = (int8_t)0x5a;
  }
  for (int i = 0; i < PADDED_IM2COL_ELEMS; ++i) {
    actual_im2col[i] = (int8_t)0x5a;
  }
  for (int i = 0; i < TRANSPOSE_SOURCE_ELEMS; ++i) {
    transpose_source[i] = 0;
  }
  for (int i = 0; i < KERNEL_ELEMS; ++i) {
    transpose_source[i] = kernel_flat_1x9[i];
  }
  for (int i = 0; i < PADDED_TRANSPOSE_ELEMS; ++i) {
    actual_transpose[i] = (int8_t)0x5a;
  }
  for (int i = 0; i < PADDED_GEMM_ELEMS; ++i) {
    zero_gemm[i] = 0;
    actual_gemm[i] = (int32_t)0x5a5a5a5a;
  }
  for (int i = 0; i < INT8_LANES; ++i) {
    requant_substitute[i] = (i < NUM_WINDOWS) ? expected_requant_output[i] : 0;
  }
  for (int i = 0; i < NUM_WINDOWS; ++i) {
    actual_final[i] = 12345.0f;
  }
}

static int check_quant(void) {
  for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
    if (actual_quant[i] != expected_quant_input[i]) {
      printf("FAIL stage=fp32_to_int8 index=%d row=%d col=%d expected=%d "
             "actual=%d\n",
             i, i / INPUT_W, i % INPUT_W, (int)expected_quant_input[i],
             (int)actual_quant[i]);
      return 0;
    }
  }
  printf("PASS stage=fp32_to_int8\n");
  return 1;
}

static int check_im2col(void) {
  for (int row = 0; row < NUM_WINDOWS; ++row) {
    for (int col = 0; col < INT8_LANES; ++col) {
      const int index = row * INT8_LANES + col;
      const int8_t expected = (col < KERNEL_ELEMS)
                                  ? expected_im2col_a[row * KERNEL_ELEMS + col]
                                  : 0;
      if (actual_im2col[index] != expected) {
        printf("FAIL stage=im2col index=%d row=%d col=%d expected=%d "
               "actual=%d\n",
               index, row, col, (int)expected, (int)actual_im2col[index]);
        return 0;
      }
    }
  }
  printf("PASS stage=im2col\n");
  return 1;
}

static int check_transpose(void) {
  for (int row = 0; row < KERNEL_ELEMS; ++row) {
    for (int col = 0; col < INT8_LANES; ++col) {
      const int index = row * INT8_LANES + col;
      const int8_t expected = (col == 0) ? expected_transpose_b[row] : 0;
      if (actual_transpose[index] != expected) {
        printf("FAIL stage=transpose index=%d row=%d col=%d expected=%d "
               "actual=%d\n",
               index, row, col, (int)expected, (int)actual_transpose[index]);
        return 0;
      }
    }
  }
  printf("PASS stage=transpose\n");
  return 1;
}

static int check_gemm(void) {
  for (int i = 0; i < NUM_WINDOWS; ++i) {
    const int32_t actual = actual_gemm[i * ACC_LANES];
    if (actual != expected_gemm_c[i]) {
      printf("FAIL stage=gemm index=%d row=%d col=%d expected=%d actual=%d\n",
             i, i / OUTPUT_W, i % OUTPUT_W, (int)expected_gemm_c[i],
             (int)actual);
      return 0;
    }
  }
  printf("PASS stage=gemm\n");
  return 1;
}

static int check_requant_substitute(void) {
  for (int i = 0; i < NUM_WINDOWS; ++i) {
    if (requant_substitute[i] != expected_requant_output[i]) {
      printf("FAIL stage=int32_to_int8_substitute index=%d row=%d col=%d "
             "expected=%d actual=%d\n",
             i, i / OUTPUT_W, i % OUTPUT_W, (int)expected_requant_output[i],
             (int)requant_substitute[i]);
      return 0;
    }
  }
  printf("PASS stage=int32_to_int8_substitute\n");
  return 1;
}

static int check_final(void) {
  for (int i = 0; i < NUM_WINDOWS; ++i) {
    const float error = fabsf(actual_final[i] - expected_final_fp32[i]);
    if (error > 1.0e-6f) {
      union {
        float value;
        uint32_t bits;
      } expected = {.value = expected_final_fp32[i]},
        actual = {.value = actual_final[i]};
      printf("FAIL stage=int8_to_fp32 index=%d row=%d col=%d "
             "expected_bits=0x%08x actual_bits=0x%08x\n",
             i, i / OUTPUT_W, i % OUTPUT_W, expected.bits, actual.bits);
      return 0;
    }
  }
  printf("PASS stage=int8_to_fp32\n");
  return 1;
}

static int run_test(void) {
  const uint32_t fp32_input_bank = 0;
  const uint32_t quant_bank = 1;
  const uint32_t im2col_a_bank = 2;

  bb_mem_alloc(fp32_input_bank, 1, 4);
  bb_mem_alloc(quant_bank, 1, 1);
  bb_mvin((uintptr_t)packed_input_fp32, fp32_input_bank, QUANT_ROWS, 1);
  bb_fp2int(fp32_input_bank, quant_bank, QUANT_ROWS, SCALE_FP32_TO_INT8);
  bb_mvout((uintptr_t)actual_quant, quant_bank, QUANT_ROWS, 1);
  bb_fence();
  if (!check_quant()) {
    return 0;
  }
  bb_mem_release(fp32_input_bank);

  bb_mem_alloc(im2col_a_bank, 1, 1);
  bb_im2col(quant_bank, im2col_a_bank, INPUT_H, KERNEL_H, STRIDE, PADDING);
  bb_mvout((uintptr_t)actual_im2col, im2col_a_bank, IM2COL_ROWS, 1);
  bb_fence();
  if (!check_im2col()) {
    return 0;
  }
  bb_mem_release(quant_bank);

  const uint32_t kernel_source_bank = 0;
  const uint32_t transpose_b_bank = 1;
  bb_mem_alloc(kernel_source_bank, 1, 1);
  bb_mem_alloc(transpose_b_bank, 1, 1);
  bb_mvin((uintptr_t)transpose_source, kernel_source_bank, KERNEL_ELEMS, 1);
  bb_transpose(kernel_source_bank, transpose_b_bank, KERNEL_ELEMS, 0);
  bb_mvout((uintptr_t)actual_transpose, transpose_b_bank, KERNEL_ELEMS, 1);
  bb_fence();
  if (!check_transpose()) {
    return 0;
  }
  bb_mem_release(kernel_source_bank);

  const uint32_t gemm_c_bank = 3;
  bb_mem_alloc(gemm_c_bank, 1, 4);
  bb_mvin((uintptr_t)zero_gemm, gemm_c_bank, NUM_WINDOWS, 1);
  bb_matrix_mnk(im2col_a_bank, transpose_b_bank, gemm_c_bank, NUM_WINDOWS, 1,
                KERNEL_ELEMS);
  bb_mvout((uintptr_t)actual_gemm, gemm_c_bank, NUM_WINDOWS, 1);
  bb_fence();
  if (!check_gemm()) {
    return 0;
  }
  bb_mem_release(im2col_a_bank);
  bb_mem_release(transpose_b_bank);
  bb_mem_release(gemm_c_bank);

  /*
   * TODO(NPU): Replace this substitute with the future direct INT32->INT8
   * requantization instruction:
   *
   *   bb_int32_to_int8(gemm_c_bank, requant_bank, NUM_WINDOWS,
   *                    SCALE_INT32_TO_INT8);
   *
   * No such ISA/API currently exists. After the hardware GEMM result has been
   * checked against expected_gemm_c, the fixed expected_requant_output is
   * uploaded as the permitted temporary substitute for the next NPU stage.
   */
  if (!check_requant_substitute()) {
    return 0;
  }

  const uint32_t requant_bank = 0;
  const uint32_t final_fp32_bank = 1;
  bb_mem_alloc(requant_bank, 1, 1);
  bb_mem_alloc(final_fp32_bank, 1, 4);
  bb_mvin((uintptr_t)requant_substitute, requant_bank, 1, 1);
  bb_int2fp(requant_bank, final_fp32_bank, 1, SCALE_INT8_TO_FP32);
  bb_mvout((uintptr_t)actual_final, final_fp32_bank, 1, 1);
  bb_fence();
  if (!check_final()) {
    return 0;
  }

  bb_mem_release(requant_bank);
  bb_mem_release(final_fp32_bank);
  return 1;
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  initialize_buffers();
  const int passed = run_test();
  printf("conv1layer_quant_im2col_transpose_gemm_test: %s\n",
         passed ? "PASS" : "FAIL");

#ifdef MULTICORE
  exit(passed ? 0 : 1);
#endif
  return passed ? 0 : 1;
}
