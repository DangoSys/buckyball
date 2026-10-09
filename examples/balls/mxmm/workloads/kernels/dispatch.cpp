#include "norm_window_image.h"
#include "window.h"
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <params.h>

extern "C" void
_mlir_ciface_rvv_norm_window(uint64_t readBank, uint64_t writeBank,
                             uint64_t phase, uint64_t width, uint64_t columns,
                             uint64_t count, uint64_t full, uint64_t start,
                             uint64_t first, uint64_t last, uint64_t bankBytes,
                             uint32_t meanBits, uint32_t epsilonBits) {
  constexpr unsigned programBank = VIRTUAL_BANK_NUM;
  if (phase == 2) {
    release_kernel(programBank);
    return;
  }
  if (phase > 1 || !width || width > bankBytes / sizeof(float)) {
    fputs("MXMM norm/window requires a complete row within its bank\n", stderr);
    abort();
  }
  if (phase == 0)
    mvin_kernel(norm_window_image, sizeof(norm_window_image), programBank);
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding) : : "memory");
  struct alignas(16) Packet {
    kernel_launch launch;
    Window parameters;
    uint32_t padding;
  } packet{{norm_window_entry,
            norm_window_text,
            0x40002000,
            {(uint64_t(2) << 32) + sizeof(kernel_launch),
             (uint64_t(2) << 32) + offsetof(kernel_launch, reserved)},
            0},
           {uint32_t(phase), uint32_t(width), meanBits, epsilonBits,
            uint32_t(columns), uint32_t(count), uint32_t(full), uint32_t(start),
            uint32_t(first), uint32_t(last), rounding << 5},
           0};
  bb_mvin_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
  // Finish descriptor DMA before this stack packet can be reused.
  bb_fence();
  run_kernel(readBank, programBank, writeBank, 0);
  if (phase == 1)
    return;
  alignas(16) kernel_launch completed;
  bb_mvout_group((uintptr_t)&completed, writeBank, 0, sizeof(completed) / 16,
                 1);
  bb_fence();
  if (std::memcmp(&packet.launch, &completed,
                  offsetof(kernel_launch, reserved)) ||
      (completed.reserved & ~UINT64_C(31))) {
    fputs(
        "MXMM norm/window returned an invalid descriptor or exception flags\n",
        stderr);
    abort();
  }
  asm volatile("csrs fflags, %0" : : "r"(completed.reserved) : "memory");
}
