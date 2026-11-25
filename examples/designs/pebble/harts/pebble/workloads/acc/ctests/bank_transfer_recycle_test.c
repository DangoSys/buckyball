#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t source[2] __attribute__((aligned(64)));
static uint64_t target[2] __attribute__((aligned(64)));
static uint64_t output[4] __attribute__((aligned(64)));

int main(void) {
  for (unsigned iteration = 0; iteration < 128; ++iteration) {
    source[0] = 0x1234000000000000ULL + iteration;
    source[1] = ~source[0];
    target[0] = 0x5678000000000000ULL + iteration;
    target[1] = ~target[0];
    bb_mem_alloc(0, 1, 1);
    bb_mem_alloc(1, 1, 1);
    bb_mvin((uintptr_t)source, 0, 1, 1);
    bb_mvin((uintptr_t)target, 1, 1, 1);
    bb_mem_transfer(0, 1);
    bb_mvout((uintptr_t)output, 1, 1, 1);
    if (output[0] != target[0] || output[1] != target[1] ||
        output[2] != source[0] || output[3] != source[1]) {
      printf("bank transfer recycle mismatch iteration=%u\n", iteration);
      return 1;
    }
    bb_mem_release(1);
  }
  puts("Bank transfer recycle PASS");
  return 0;
}
