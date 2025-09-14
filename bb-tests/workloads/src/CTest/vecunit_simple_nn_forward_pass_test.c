#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

// 定义神经网络参数
#define INPUT_SIZE DIM
#define HIDDEN_SIZE DIM
#define OUTPUT_SIZE DIM

// 测试矩阵和数据缓冲区
static elem_t input_data[DIM * DIM] __attribute__((aligned(64)));
static elem_t weights1[HIDDEN_SIZE * INPUT_SIZE] __attribute__((aligned(64)));
static elem_t weights2[OUTPUT_SIZE * HIDDEN_SIZE] __attribute__((aligned(64)));
static result_t hidden_output[DIM * DIM] __attribute__((aligned(64)));
static result_t final_output[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_output[DIM * DIM] __attribute__((aligned(64)));

// ReLU激活函数（在CPU上执行）
void relu(result_t *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    if (matrix[i] < 0) {
      matrix[i] = 0;
    }
  }
}

// 量化函数（将int32结果量化为elem_t类型）
void quantize_matrix(result_t *src, elem_t *dst, int size) {
  for (int i = 0; i < size * size; i++) {
    dst[i] = (src[i] > 127) ? 127 : (src[i] < -128) ? -128 : (elem_t)src[i];
  }
}

// CPU上的神经网络前向传播
void cpu_nn_forward(elem_t *input, elem_t *w1, elem_t *w2, result_t *hidden,
                    result_t *output, int size) {
  // 输入层 -> 隐藏层
  cpu_matmul(input, w1, hidden, size, size, size);
  relu(hidden, size, size); // 应用ReLU激活

  // 量化隐藏层输出作为下一层输入
  static elem_t hidden_quantized[DIM * DIM];
  quantize_matrix(hidden, hidden_quantized, size);

  // 隐藏层 -> 输出层
  cpu_matmul(hidden_quantized, w2, output, size, size, size);
}

// 执行硬件矩阵乘法
void hw_matmul(elem_t *a, elem_t *b, result_t *c, int size) {
  // 转置左矩阵
  static elem_t a_transposed[DIM * DIM] __attribute__((aligned(64)));
  transpose_u8_matrix(a, a_transposed, size, size);

  // 移动矩阵到暂存器
  uint32_t op1_addr = spad_addr(0, 0); // spad0: 操作数A, 偏移0
  uint32_t op2_addr = spad_addr(1, 0); // spad1: 操作数B, 偏移0
  uint32_t wr_addr = spad_addr(4, 0);  // acc0: 写入累加器, 偏移0

  bb_mvin((uintptr_t)a_transposed, op1_addr, size);
  bb_mvin((uintptr_t)b, op2_addr, size);
  bb_mvin((uintptr_t)c, wr_addr, size << 2);
  bb_fence();

  // 执行矩阵乘法
  bb_mul_warp16(op1_addr, op2_addr, wr_addr, size);
  bb_fence();

  // 移回结果
  bb_mvout((uintptr_t)c, wr_addr, size << 2);
  bb_fence();
}

void hw_nn_forward(elem_t *input, elem_t *w1, elem_t *w2, result_t *hidden,
                   result_t *output, int size) {
  // 输入层 -> 隐藏层
  hw_matmul(input, w1, hidden, size);
  relu(hidden, size, size); // 在CPU上应用ReLU

  // 量化隐藏层输出作为下一层输入
  static elem_t hidden_quantized[DIM * DIM];
  quantize_matrix(hidden, hidden_quantized, size);

  // 隐藏层 -> 输出层
  hw_matmul(hidden_quantized, w2, output, size);
}

// 执行神经网络测试
int test_neural_network() {
  // 初始化数据
  printf("Initializing random input data and weights...\n");
  init_u8_random_matrix(input_data, DIM, DIM, 123);

  // 初始化权重
  srand(114);
  for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
    weights1[i] = rand() % 128;
  }
  srand(514);
  for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
    weights2[i] = rand() % 128;
  }

  // 清空输出缓冲区
  clear_u32_matrix(hidden_output, DIM, DIM);
  clear_u32_matrix(expected_output, DIM, DIM);

  // 在CPU上生成预期结果
  printf("Running CPU Neural Network Forward Pass...\n");
  cpu_nn_forward(input_data, weights1, weights2, hidden_output, expected_output,
                 DIM);

  // 重新清空hidden_output用于硬件计算
  clear_u32_matrix(hidden_output, DIM, DIM);
  clear_u32_matrix(final_output, DIM, DIM);

  // 在硬件上执行神经网络前向传播
  printf("Running Hardware Neural Network Forward Pass...\n");
  hw_nn_forward(input_data, weights1, weights2, hidden_output, final_output,
                DIM);

  // 比较硬件输出和预期输出
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
    printf("Neural Network Test PASSED\n");
  } else {
    printf("Neural Network Test FAILED\n");
  }

#ifdef MULTICORE
  exit(0);
#endif
  return 0;
}
