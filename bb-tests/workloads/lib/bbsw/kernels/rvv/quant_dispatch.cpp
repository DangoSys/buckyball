#include "../mxfp8/quant.h"
#include "images.h"
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void mxfp8_quant(const uint32_t *input, uint8_t *output,
                            uint32_t count) {
  constexpr unsigned readBank = 4, writeBank = 5;
  constexpr unsigned programBank = VIRTUAL_BANK_NUM;
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(2, readBank);
  bb_mem_transfer(0, writeBank);
  bb_mem_transfer(1, writeBank);
  mvin_kernel(images::quant.bytes, images::quant.size, programBank);
  bb_mvin_group((uintptr_t)input, readBank, 0, count / 4, 1);
  alignas(16) uint8_t descriptor[sizeof(kernel_launch)]{};
  kernel_launch call{images::quant.entry,
                     images::quant.text_bytes,
                     0x40002000,
                     {uint64_t(2) << 32, (uint64_t(2) << 32) + count, 0, count,
                      (uint64_t(1) << 32) + offsetof(kernel_launch, reserved)},
                     0};
  std::memcpy(descriptor, &call, sizeof(call));
  bb_mvin_group((uintptr_t)descriptor, writeBank, 0, sizeof(descriptor) / 16,
                1);
  bb_fence();
  run_kernel(readBank, programBank, writeBank, 0);
  bb_fence();
  bb_mvout_group((uintptr_t)descriptor, writeBank, 0, sizeof(descriptor) / 16,
                 1);
  bb_mvout_group((uintptr_t)output, writeBank, 1,
                 (count + count / 32 + 15) / 16, 1);
  bb_fence();
  if (descriptor[offsetof(kernel_launch, reserved)]) {
    fputs("mxfp8: non-finite activation\n", stderr);
    abort();
  }
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}
