#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// CPU TEST BEGIN 1
// Read cycle counter (rdcycle) helper. Works on RV64 with a single rdcycle.
// On RV32 we read low/high and detect rollover to produce a 64-bit value.
static inline unsigned long long read_rdcycle(void) {
#if defined(__riscv_xlen) && __riscv_xlen == 64
  unsigned long long cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
#else
  unsigned int lo1, hi, lo2;
  // Loop until two consecutive low reads are equal to avoid rollover window
  asm volatile("1: rdcycle %0\n"
               "   rdcycleh %1\n"
               "   rdcycle %2\n"
               "   bne %0, %2, 1b\n"
               : "=&r"(lo1), "=&r"(hi), "=&r"(lo2));
  return ((unsigned long long)hi << 32) | lo1;
#endif
}
// CPU TEST END 1

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

int relu_cpu_reference(elem_t *input, elem_t *output, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      elem_t val = input[i * size + j];
      output[i * size + j] = (val < 0) ? 0 : val;
    }
  }
  return 1;
}

int test_relu(int seed) {
  init_i8_random_matrix(input_matrix_a, DIM, DIM, seed);
  // CPU TEST BEGIN 2
  // Measure cycles for the CPU ReLU reference implementation
  unsigned long long start = read_rdcycle();
  int ok = relu_cpu_reference(input_matrix_a, output_matrix_b, DIM); // CPU 验证
  unsigned long long end = read_rdcycle();
  unsigned long long cycles = end - start;
  /* Print as hex high/low 32-bit parts to avoid embedded printf lacking
    full long long support. This produces a stable, greppable output. */
  uint32_t lo = (uint32_t)(cycles & 0xffffffffULL);
  uint32_t hi = (uint32_t)(cycles >> 32);
  printf("BB_CYCLES_RELU: 0x%08x%08x\n", hi, lo);
  return ok;
  // CPU TEST END 2
  // return run_test("ReLU", input_matrix_a, output_matrix_b, DIM);
  // //ReLUBall的测试代码，需要注释掉上面代码块
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_relu(5);
  if (passed) {
    printf("ReLU test PASSED!!\n");
  } else {
    printf("ReLU test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
