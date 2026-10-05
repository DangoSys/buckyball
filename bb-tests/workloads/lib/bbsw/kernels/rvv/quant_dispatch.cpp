#include "../mxfp8/quant.h"
#include "images.h"
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void mxfp8_quant(const uint32_t *input, uint8_t *output,
                            uint32_t count) {
  constexpr unsigned outputBank = 1, inputBank = 2;
  for (unsigned bank = 0; bank <= inputBank; ++bank)
    bb_mem_alloc(bank, 1, 1);
  mvin_kernel(images::quant.bytes, images::quant.size, 0);
  bb_mvin((uintptr_t)input, inputBank, count / 4, 1);
  alignas(16) uint8_t descriptor[64]{};
  kernel_launch call{images::quant.entry,
                     images::quant.text_bytes,
                     0x80002000,
                     {outputBank << 16, inputBank << 16, count, 48},
                     0};
  std::memcpy(descriptor, &call, sizeof(call));
  bb_mvin((uintptr_t)descriptor, 0, sizeof(descriptor) / 16, 1);
  bb_fence();
  run_kernel(0, 0);
  bb_fence();
  bb_mvout((uintptr_t)descriptor, 0, sizeof(descriptor) / 16, 1);
  bb_mvout((uintptr_t)output, outputBank, (count + count / 32 + 15) / 16, 1);
  bb_fence();
  if (descriptor[48]) {
    fputs("mxfp8: non-finite activation\n", stderr);
    abort();
  }
  for (unsigned bank = 0; bank <= inputBank; ++bank)
    bb_mem_release(bank);
}
