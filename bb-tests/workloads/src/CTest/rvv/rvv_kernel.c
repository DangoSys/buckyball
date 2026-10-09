#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

// Deliberately start the image four bytes into an aligned object.
static const uint32_t image[] __attribute__((aligned(16))) = {
    0x00000000, 0x31564b52, 0x00000030, 0x00000000, 0x40000000,
    0x00000004, 0x00000000, 0x400002b7, 0x0002a303, 0x010673d7,
    0x0205e087, 0x02134157, 0x02056127, 0x00239413, 0x00850533,
    0x008585b3, 0x40760633, 0xfe0610e3, 0x00008067, 0x00000003};
static uint32_t input[128] __attribute__((aligned(16)));
static uint32_t output[128] __attribute__((aligned(16)));

int main(void) {
  for (unsigned i = 0; i < 128; ++i)
    input[i] = i * 7;
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, bank == 0 ? 1 : 2);
  struct kernel_launch call __attribute__((aligned(16))) = {
      0, 48, 0x40002000, {((uint64_t)3 << 32), 0, 64}, 0};
  bb_mvin((uintptr_t)input, 2, 16, 1);
  bb_mvin((uintptr_t)&call, 0, sizeof(call) / 16, 1);
  bb_mem_transfer(2, 4);
  bb_mem_transfer(0, 5);
  bb_mem_transfer(1, 5);
  mvin_kernel(image + 1, sizeof(image) - 4, VIRTUAL_BANK_NUM);
  run_kernel(4, VIRTUAL_BANK_NUM, 5, 0);
  call.args[0] = (uint64_t)4 << 32;
  call.args[1] = (uint64_t)1 << 32;
  bb_mvin_group((uintptr_t)&call, 5, 0, sizeof(call) / 16, 1);
  run_kernel(4, VIRTUAL_BANK_NUM, 5, 0);
  uint32_t groups[2][64] __attribute__((aligned(16)));
  bb_mvout_group((uintptr_t)groups[0], 5, 1, 16, 1);
  bb_mvout_group((uintptr_t)groups[1], 5, 2, 16, 1);
  release_kernel(VIRTUAL_BANK_NUM);
  bb_fence();
  bb_mem_release(4);
  bb_mem_release(5);
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned group = 0; group < 2; ++group)
      for (unsigned lane = 0; lane < 4; ++lane)
        output[row * 8 + group * 4 + lane] = groups[group][row * 4 + lane];
  for (unsigned i = 0; i < 128; ++i)
    if (output[i] != input[i] + 3)
      return 1;
  printf("RVV KERNEL PASSED\n");
  return 0;
}
