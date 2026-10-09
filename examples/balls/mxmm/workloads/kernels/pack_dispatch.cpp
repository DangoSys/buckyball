#include "packmm.h"
#include "packmm_image.h"
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <params.h>

extern "C" void _mlir_ciface_rvv_packmm(uint64_t readBank, uint64_t writeBank,
                                        uint64_t phase, uint64_t fullK,
                                        uint64_t fullN, uint64_t startK,
                                        uint64_t startN, uint64_t countK,
                                        uint64_t columns, uint64_t rows,
                                        uint64_t first, uint64_t last,
                                        uint64_t fused) {
  constexpr unsigned programBank = VIRTUAL_BANK_NUM;
  if (phase == 0) {
    mvin_kernel(packmm_image, sizeof(packmm_image), programBank);
    return;
  }
  if (phase == 2) {
    release_kernel(programBank);
    return;
  }
  if (phase != 1) {
    fputs("MXMM pack/matrix requires load, execute, or release\n", stderr);
    abort();
  }
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding) : : "memory");
  struct alignas(16) Packet {
    kernel_launch launch;
    PackMM parameters;
    uint32_t padding;
  } packet{{packmm_entry,
            packmm_text,
            0x40002000,
            {(uint64_t(2) << 32) + sizeof(kernel_launch),
             (uint64_t(2) << 32) + offsetof(kernel_launch, reserved)},
            0},
           {uint32_t(fullK), uint32_t(fullN), uint32_t(startK),
            uint32_t(startN), uint32_t(countK), uint32_t(columns),
            uint32_t(rows), uint32_t(first), uint32_t(last), uint32_t(fused),
            rounding << 5},
           0};
  bb_mvin_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
  // Finish descriptor DMA before this stack packet can be reused.
  run_kernel(readBank, programBank, writeBank, 0);
}
