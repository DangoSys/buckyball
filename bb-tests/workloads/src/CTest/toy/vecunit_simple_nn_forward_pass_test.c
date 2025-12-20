#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM (BANK_WIDTH / sizeof(elem_t))

// Define neural network parameters
#define INPUT_SIZE DIM
#define HIDDEN_SIZE DIM
#define OUTPUT_SIZE DIM

// Test matrices and data buffers
static elem_t input_data[DIM * DIM] __attribute__((aligned(64)));
static elem_t weights1[HIDDEN_SIZE * INPUT_SIZE] __attribute__((aligned(64)));
static elem_t weights2[OUTPUT_SIZE * HIDDEN_SIZE] __attribute__((aligned(64)));
static result_t hidden_output[DIM * DIM] __attribute__((aligned(64)));
static result_t final_output[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_output[DIM * DIM] __attribute__((aligned(64)));

// ReLU activation function (executed on CPU)
void relu(result_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (matrix[i] < 0) {
      matrix[i] = 0;
    }
  }
}

// Quantization function (quantize int32 results to elem_t type)
void quantize_matrix(result_t *src, elem_t *dst, int size) {
  for (int i = 0; i < size * size; i++) {
    dst[i] = (src[i] > 127) ? 127 : (src[i] < -128) ? -128 : (elem_t)src[i];
  }
}

// Neural network forward propagation on CPU
void cpu_nn_forward(elem_t *input, elem_t *w1, elem_t *w2, result_t *hidden,
                    result_t *output, int size) {
  // Input layer -> hidden layer
  cpu_matmul(input, w1, hidden, size, size, size);
  // Apply ReLU activation
  relu(hidden, size, size);

  // Quantize hidden layer output as input for next layer
  static elem_t hidden_quantized[DIM * DIM];
  quantize_matrix(hidden, hidden_quantized, size);

  // Hidden layer -> output layer
  cpu_matmul(hidden_quantized, w2, output, size, size, size);
}

// Execute hardware matrix multiplication
void hw_matmul(elem_t *a, elem_t *b, result_t *c, int size) {
  // Transpose left matrix
  static elem_t a_transposed[DIM * DIM] __attribute__((aligned(64)));
  transpose_u8_matrix(a, a_transposed, size, size);

  // Move matrices to scratchpad
  // spad0: operand A, offset 0
  uint32_t op1_bank_id = 0;
  // spad1: operand B, offset 0
  uint32_t op2_bank_id = 1;
  // acc0: write to accumulator, offset 0
  int acc_bank_id = bb_mset(0, 0, 1, 4, 1, 4);

  bb_mvin((uintptr_t)a_transposed, op1_bank_id, size, 1);
  bb_mvin((uintptr_t)b, op2_bank_id, size, 1);
  bb_mvin((uintptr_t)c, acc_bank_id, size << 2, 1);
  bb_fence();

  // Execute matrix multiplication
  bb_mul_warp16(op1_bank_id, op2_bank_id, acc_bank_id, size, 0);
  bb_fence();

  // Move result back
  bb_mvout((uintptr_t)c, acc_bank_id, size << 2, 1);
  bb_fence();
}

void hw_nn_forward(elem_t *input, elem_t *w1, elem_t *w2, result_t *hidden,
                   result_t *output, int size) {
  // Input layer -> hidden layer
  hw_matmul(input, w1, hidden, size);
  // Apply ReLU on CPU
  relu(hidden, size, size);

  // Quantize hidden layer output as input for next layer
  static elem_t hidden_quantized[DIM * DIM];
  quantize_matrix(hidden, hidden_quantized, size);

  // Hidden layer -> output layer
  hw_matmul(hidden_quantized, w2, output, size);
}

// Execute neural network test
int test_neural_network() {
  // Initialize data
  printf("Initializing random input data and weights...\n");
  init_u8_random_matrix(input_data, DIM, DIM, 123);

  // Initialize weights
  srand(114);
  for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
    weights1[i] = rand() % 128;
  }
  srand(514);
  for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
    weights2[i] = rand() % 128;
  }

  // Clear output buffers
  clear_u32_matrix(hidden_output, DIM, DIM);
  clear_u32_matrix(expected_output, DIM, DIM);

  // Generate expected results on CPU
  printf("Running CPU Neural Network Forward Pass...\n");
  cpu_nn_forward(input_data, weights1, weights2, hidden_output, expected_output,
                 DIM);

  // Clear hidden_output again for hardware computation
  clear_u32_matrix(hidden_output, DIM, DIM);
  clear_u32_matrix(final_output, DIM, DIM);

  // Execute neural network forward propagation on hardware
  printf("Running Hardware Neural Network Forward Pass...\n");
  hw_nn_forward(input_data, weights1, weights2, hidden_output, final_output,
                DIM);

  // Compare hardware output with expected output
  printf("Comparing hardware output with expected output...\n");
  if (compare_u32_matrices(final_output, expected_output, DIM, DIM)) {
    return 1;
  } else {
    return 0;
  }
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  printf("Neural Network Test Starting...\n");
  int pass = test_neural_network();
  if (pass) {
    printf("Neural Network test PASSED\n");
    return 0;
  } else {
    printf("Neural Network test FAILED\n");
    return 1;
  }

#ifdef MULTICORE
  exit(0);
#endif
  return 0;
}
