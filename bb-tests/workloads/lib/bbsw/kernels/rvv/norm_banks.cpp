#include "images.h"
#include <bbhw/isa/isa.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void
_mlir_ciface_rvv_norm_banks(uint64_t descriptorBank, uint64_t outputBank,
                            uint64_t inputBank, uint64_t weightBank,
                            uint64_t width, uint64_t bankBytes,
                            uint32_t meanBits, uint32_t epsilonBits) {
  if (!width || width > bankBytes / sizeof(float) || bankBytes > 65536) {
    fputs("RVV bank norm requires a complete row within its bank\n", stderr);
    abort();
  }
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding) : : "memory");
  const uint32_t descriptorAddress = uint32_t(descriptorBank << 16);
  alignas(16) kernel_launch packet{
      images::norm.entry,
      images::norm.text_bytes,
      0x80002000,
      {uint32_t(outputBank << 16), uint32_t(inputBank << 16),
       uint32_t(weightBank << 16), uint32_t(width), meanBits, epsilonBits,
       descriptorAddress + 44, rounding << 5},
      0};
  mvin_kernel(images::norm.bytes, images::norm.size, 0);
  bb_mvin((uintptr_t)&packet, descriptorBank, sizeof(packet) / 16, 1);
  bb_fence();
  run_kernel(descriptorAddress, 0);
  bb_fence();
  alignas(16) kernel_launch completed;
  bb_mvout((uintptr_t)&completed, descriptorBank, sizeof(completed) / 16, 1);
  bb_fence();
  if (std::memcmp(&packet, &completed, 44) || (completed.reserved & ~31u)) {
    fputs("RVV bank norm returned an invalid descriptor or exception flags\n",
          stderr);
    abort();
  }
  asm volatile("csrs fflags, %0" : : "r"(completed.reserved) : "memory");
}
