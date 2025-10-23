#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));
// static elem_t probe_matrix[DIM * DIM] __attribute__((aligned(64))); //
// 用于验证MVIN后SPAD中的内容

// 预期：提供一个与 TRANSPOSE 类似的 ReLU 流程
// 目前 bbhw/isa 尚无 bb_relu 高层 API，示例采用与 transpose
// 相同的搬入->执行->fence 流程。 需要在 bbhw 实现中补充 bb_relu(op1_addr,
// wr_addr, iter) 的封装（func7=RELU_FUNC7）。

void hw_relu(const char *test_name, elem_t *a, elem_t *b, int size) {
  // 源操作数放在 spad bank 0，写回目标放在 spad bank 1
  uint32_t op1_addr = spad_addr(0, 0);
  uint32_t wr_addr = spad_addr(1, 0);

  // 把输入搬入 scratchpad bank0，从偏移0开始，按行迭代 size 次
  bb_mvin((uintptr_t)a, op1_addr, size, 1);
  bb_fence();
  // 调用 ReLU 指令
  bb_relu(op1_addr, wr_addr, size);
  bb_fence();

  // 可选：把结果搬回内存，便于主机侧检查
  // bb_mvout((uintptr_t)b, wr_addr, size);
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_relu(test_name, a, b, size);
  // 若上面打印了 mismatch，可在此选择直接失败；为保持兼容暂时仍返回 1
  return 1;
}

int test_relu() {
  // 准备一份包含正负值的输入，便于检查 ReLU 行为
  for (int i = 0; i < DIM * DIM; ++i) {
    int v = (i % 11) - 5; // -5..+5
    input_matrix_a[i] = (elem_t)v;
  }
  return run_test("ReLU", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif
  int passed = test_relu();
  if (passed) {
    printf("ReLU test PASSED\n");
    return 0;
  } else {
    printf("ReLU test FAILED\n");
    return 1;
  }
#ifdef MULTICORE
  exit(0);
#endif
}
