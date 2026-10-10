#include "packmm.h"
#include <ballISA.h>
#include <bbsw/kernels/rvv/kernels.h>

extern "C" void rvv_pack(uint32_t *, const uint32_t *, size_t, size_t, size_t,
                         size_t, size_t, size_t);

extern "C" void packmm(const PackMM *parameters, uint32_t *flags) {
  asm volatile("csrw fcsr, %0" : : "r"(parameters->rounding) : "memory");
  rvv_pack((uint32_t *)(uint64_t(3) << 32), (const uint32_t *)0,
           parameters->fullK, parameters->fullN, parameters->startK,
           parameters->startN, parameters->countK, parameters->columns);
  const uint64_t banks = UINT64_C(1) | (UINT64_C(3) << 10) |
                         (UINT64_C(4) << 20) |
                         (uint64_t(parameters->countK) << 30);
  const uint64_t config =
      uint64_t(parameters->rows) | (uint64_t(parameters->columns) << 12) |
      (uint64_t(parameters->first) << 24) | (uint64_t(parameters->last) << 25);
  if (parameters->fused)
    asm volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"
                 :
                 : "r"(banks), "r"(config), "i"(BB_FUNC7_MXMM_FMA32)
                 : "memory");
  else
    asm volatile(".insn r 0x7b, 3, %c2, x0, %0, %1"
                 :
                 : "r"(banks), "r"(config), "i"(BB_FUNC7_MXMM_F32)
                 : "memory");
  uint64_t result;
  asm volatile("csrr %0, fflags" : "=r"(result) : : "memory");
  *flags = uint32_t(result);
}
