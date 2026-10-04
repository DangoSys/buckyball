#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

// Deliberately start the image four bytes into an aligned object.
static const uint32_t image[] __attribute__((aligned(16))) = {
    0,          0x31564b52, 28,         0,          0x80000000,
    4,          0,          0x800002b7, 0x0002a303, 0x010673d7,
    0x0205e087, 0x02134157, 0x02056127, 0x00008067, 3};
static uint32_t input[16] __attribute__((aligned(16)));
static uint32_t output[16] __attribute__((aligned(16)));

int main(void) {
  for (unsigned i = 0; i < 16; ++i)
    input[i] = i * 7;
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  struct kernel_launch call __attribute__((aligned(16))) = {
      0, 28, 0x80002000, {1 << 16, 2 << 16, 16}, 0};
  bb_mvin((uintptr_t)input, 2, 4, 1);
  bb_mvin((uintptr_t)&call, 0, 3, 1);
  mvin_kernel(image + 1, sizeof(image) - 4, 0);
  run_kernel(0, 0);
  bb_mvout((uintptr_t)output, 1, 4, 1);
  bb_fence();
  for (unsigned i = 0; i < 16; ++i)
    if (output[i] != input[i] + 3)
      return 1;
  printf("RVV KERNEL PASSED\n");
  return 0;
}
