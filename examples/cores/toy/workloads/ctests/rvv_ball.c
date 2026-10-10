#include <bbhw/isa/isa.h>
#include <params.h>
#include <stdint.h>
#include <stdio.h>

static const uint32_t image[] __attribute__((aligned(16))) = {
    0x31564b52, 12, 0, 0x40000000, 0, 0, 0x64b5307b, 0x64d6307b, 0x00008067};
static int32_t input[128] __attribute__((aligned(16)));
static int32_t output[128] __attribute__((aligned(16)));

int main(void) {
  for (unsigned i = 0; i < 128; ++i)
    input[i] = ((int)i - 64) * 7;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(2, 1, 2);
  struct kernel_launch call
      __attribute__((aligned(16))) = {0,
                                      12,
                                      0x40002000,
                                      {BB_BANK0(2) | BB_ITER(16), BANK_LINES,
                                       BB_BANK0(3) | BB_ITER(16), BANK_LINES},
                                      0};
  bb_mvin((uintptr_t)input, 2, 16, 1);
  bb_mvin((uintptr_t)&call, 0, sizeof(call) / 16, 1);
  bb_mem_alloc(4, 1, 1);
  bb_mem_transfer(0, 3);
  bb_mem_transfer(2, 3);
  mvin_kernel(image, sizeof(image), VIRTUAL_BANK_NUM);
  run_kernel(4, VIRTUAL_BANK_NUM, 3, 0);
  int32_t groups[2][64] __attribute__((aligned(16)));
  bb_mvout_group((uintptr_t)groups[0], 3, 1, 16, 1);
  bb_mvout_group((uintptr_t)groups[1], 3, 2, 16, 1);
  release_kernel(VIRTUAL_BANK_NUM);
  for (unsigned row = 0; row < 16; ++row)
    for (unsigned group = 0; group < 2; ++group)
      for (unsigned lane = 0; lane < 4; ++lane)
        output[row * 8 + group * 4 + lane] = groups[group][row * 4 + lane];
  for (unsigned i = 0; i < 128; ++i)
    if (output[i] != (input[i] > 0 ? input[i] : 0))
      return 1;
  bb_mem_release(3);
  bb_mem_release(4);
  puts("RVV BALL PASSED");
  return 0;
}
