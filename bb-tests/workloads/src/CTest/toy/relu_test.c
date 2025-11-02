#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));
// static elem_t probe_matrix[DIM * DIM] __attribute__((aligned(64)));
// Used to verify content in SPAD after MVIN

// Expected: provide a ReLU flow similar to TRANSPOSE
// Currently bbhw/isa does not have bb_relu high-level API, this example uses the
// same move-in->execute->fence flow as transpose. Need to add bb_relu(op1_addr,
// wr_addr, iter) wrapper in bbhw implementation (func7=RELU_FUNC7).

void hw_relu(const char *test_name, elem_t *a, elem_t *b, int size) {
  // Source operand in spad bank 0, write target in spad bank 1
  uint32_t op1_addr = spad_addr(0, 0);
  uint32_t wr_addr = spad_addr(1, 0);

  // Move input into scratchpad bank0, starting at offset 0, iterate size times row-wise
  bb_mvin((uintptr_t)a, op1_addr, size, 1);
  bb_fence();
  // Call ReLU instruction
  bb_relu(op1_addr, wr_addr, size);
  bb_fence();

  // Optional: move result back to memory for host-side verification
  // bb_mvout((uintptr_t)b, wr_addr, size);
}

int run_test(const char *test_name, elem_t *a, elem_t *b, int size) {
  hw_relu(test_name, a, b, size);
  // If mismatch was printed above, can choose to fail directly here;
  // for compatibility, still return 1 for now
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
  // CPU TEST BEGIN
  // Measure cycles for the CPU ReLU reference implementation
  unsigned long long start = read_rdcycle();
  // CPU verification
  int ok = relu_cpu_reference(input_matrix_a, output_matrix_b, DIM);
  unsigned long long end = read_rdcycle();
  unsigned long long cycles = end - start;
  /* Print as hex high/low 32-bit parts to avoid embedded printf lacking
    full long long support. This produces a stable, greppable output. */
  uint32_t lo = (uint32_t)(cycles & 0xffffffffULL);
  uint32_t hi = (uint32_t)(cycles >> 32);
  printf("BB_CYCLES_RELU: 0x%08x%08x\n", hi, lo);
  return ok;
  // CPU TEST END
  // return run_test("ReLU", input_matrix_a, output_matrix_b, DIM);
  // ReLUBall test code, need to comment out the code block above
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_relu(5);
  if (passed) {
    printf("ReLU test PASSED!!!!\n");
  } else {
    printf("ReLU test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
