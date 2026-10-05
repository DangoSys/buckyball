#include "images.h"
#include <bbhw/isa/isa.h>
#include <cstdint>

extern "C" void _mlir_ciface_rvv_pack_banks(uint64_t descriptorBank,
                                            uint64_t outputBank,
                                            uint64_t inputBank, uint64_t fullK,
                                            uint64_t fullN, uint64_t startK,
                                            uint64_t startN, uint64_t countK,
                                            uint64_t columns) {
  alignas(16) kernel_launch packet{
      images::pack.entry,
      images::pack.text_bytes,
      0x80002000,
      {uint32_t(outputBank << 16), uint32_t(inputBank << 16), uint32_t(fullK),
       uint32_t(fullN), uint32_t(startK), uint32_t(startN), uint32_t(countK),
       uint32_t(columns)},
      0};
  mvin_kernel(images::pack.bytes, images::pack.size, 0);
  bb_mvin((uintptr_t)&packet, descriptorBank, sizeof(packet) / 16, 1);
  bb_fence();
  run_kernel(uint32_t(descriptorBank << 16), 0);
  bb_fence();
}
