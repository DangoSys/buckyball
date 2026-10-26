#include <bbhw/isa/isa.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t input[2] __attribute__((aligned(64)));
static uint64_t output[2] __attribute__((aligned(64)));

int main(void) {
  for (unsigned iteration = 0; iteration < 128; ++iteration) {
    input[0] = 0x1234567800000000ULL + iteration;
    input[1] = ~input[0];
    output[0] = output[1] = 0;
    bb_mem_alloc(1, 1, 1);
    bb_mvin((uintptr_t)input, 1, 1, 1);
    bb_mvout((uintptr_t)output, 1, 1, 1);
    if (output[0] != input[0] || output[1] != input[1]) {
      printf("bank recycle mismatch iteration=%u\n", iteration);
      return 1;
    }
    bb_mem_release(1);
  }
  puts("Bank recycle PASS");
  return 0;
}
