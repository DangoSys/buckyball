#include <ballISA.h>
#include <bbsw/kernels/rvv/kernels.h>
#include <stddef.h>
#include <stdint.h>

#include "window.h"

extern "C" void norm_window(const Window *parameters, uint32_t *flags) {
  asm volatile("csrw fcsr, %0" ::"r"(parameters->rounding) : "memory");
  if (parameters->phase == 0) {
    rvv_norm((float *)(uint64_t(3) << 32), (const float *)0,
             (const float *)(uint64_t(1) << 32), parameters->width,
             parameters->mean, parameters->epsilon);
    uint64_t banks =
        UINT64_C(3) | (UINT64_C(4) << 20) | (uint64_t(parameters->width) << 30);
    uint64_t zero = 0;
    asm volatile(".insn r 0x7b, 3, %c2, x0, %0, %1" ::"r"(banks), "r"(zero),
                 "i"(BB_FUNC7_MXQUANT)
                 : "memory");
  } else if (parameters->phase == 1) {
    uint64_t banks =
        UINT64_C(4) | (UINT64_C(3) << 20) | (uint64_t(parameters->count) << 30);
    uint64_t config = UINT64_C(1) | (uint64_t(parameters->columns) << 12) |
                      (uint64_t(parameters->first) << 24) |
                      (uint64_t(parameters->last) << 25) |
                      (uint64_t(parameters->full) << 32) |
                      (uint64_t(parameters->start) << 48);
    asm volatile(".insn r 0x7b, 3, %c2, x0, %0, %1" ::"r"(banks), "r"(config),
                 "i"(BB_FUNC7_MXMM_MXFP8_WINDOW)
                 : "memory");
  } else {
    asm volatile("ebreak");
  }
  uint64_t result;
  asm volatile("csrr %0, fflags" : "=r"(result)::"memory");
  *flags = uint32_t(result);
}
