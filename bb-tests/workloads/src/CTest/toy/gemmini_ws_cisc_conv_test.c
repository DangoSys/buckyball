#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>
#include <stdio.h>

// Pointwise conv: batch=1, in_dim=1, in_channels=DIM, out_channels=DIM,
// kernel=1 Degenerates to a single DIM x DIM matmul (input[1x1xIC] *
// weight[1x1xICxOC])
#define DIM 16
#define BATCH 1
#define IN_DIM 1
#define OUT_DIM 1
#define IN_CH DIM
#define OUT_CH DIM
#define KERNEL_DIM 1

// input layout: [batch, in_dim, in_dim, in_ch] = [1, 1, 1, 16] = 16 bytes
// weight layout: [krow, kcol, in_ch, out_ch] = [1, 1, 16, 16] = 256 bytes
// output layout: [batch, out_dim, out_dim, out_ch] = [1, 1, 1, 16] x 4 bytes =
// 64 bytes
static elem_t input[BATCH * IN_DIM * IN_DIM * IN_CH]
    __attribute__((aligned(64)));
static elem_t weight[KERNEL_DIM * KERNEL_DIM * IN_CH * OUT_CH]
    __attribute__((aligned(64)));
static result_t output[BATCH * OUT_DIM * OUT_DIM * OUT_CH]
    __attribute__((aligned(64)));
static result_t expected[BATCH * OUT_DIM * OUT_DIM * OUT_CH]
    __attribute__((aligned(64)));

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  printf("=== Gemmini WS CISC Loop Conv Test ===\n");

  // Initialize: input = 1x16 row vector, weight = 16x16 matrix
  init_u8_random_matrix(input, IN_CH, 1, 42);
  init_u8_random_matrix(weight, IN_CH, OUT_CH, 84);

  // CPU reference: output[0..OUT_CH] = sum_k(input[k] * weight[k][j])
  // cpu_matmul(A[M×K], B[K×N], C[M×N]) — here A=input(1×IN_CH),
  // B=weight(IN_CH×OUT_CH)
  cpu_matmul(input, weight, expected, 1, OUT_CH, IN_CH);

  // Strides:
  //   input_stride  = in_ch * elemSize = 16 * 1 = 16  (bytes per spatial step)
  //   weight_stride = in_ch * out_ch * elemSize = 16 * 16 * 1 = 256  (bytes per
  //   kernel step) output_stride = out_ch * accBytes = 16 * 4 = 64
  bb_gemmini_config(0, 0, 0, 0, 0);
  bb_gemmini_loop_conv_ws_config_1(BATCH, IN_DIM, IN_CH);
  bb_gemmini_loop_conv_ws_config_2(OUT_CH, OUT_DIM, 1, 0);
  bb_gemmini_loop_conv_ws_config_3(KERNEL_DIM, 0, 0, 0);
  bb_gemmini_loop_conv_ws_config_4(0); // no bias
  bb_gemmini_loop_conv_ws_config_5((uintptr_t)input);
  bb_gemmini_loop_conv_ws_config_6((uintptr_t)weight);
  bb_gemmini_loop_conv_ws_config_7((uintptr_t)output);
  bb_gemmini_loop_conv_ws_config_8(
      IN_CH, IN_CH * OUT_CH);                   // input_stride, weight_stride
  bb_gemmini_loop_conv_ws_config_9(OUT_CH * 4); // output_stride (accBytes=4)
  bb_gemmini_loop_conv_ws(
      0, 1, 2, 1); // bank_input=0, bank_weight=1, bank_output=2, no_bias=1
  bb_fence();

  if (compare_u32_matrices(output, expected, 1, OUT_CH)) {
    printf("Gemmini WS CISC Loop Conv Test PASSED\n");
    return 0;
  }
  printf("Gemmini WS CISC Loop Conv Test FAILED\n");
  return 1;
}
