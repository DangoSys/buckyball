#ifndef _BB_MXMM_H_
#define _BB_MXMM_H_
#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

static inline uint64_t bb_mxmm_cfg(uint32_t m, uint32_t n, uint32_t first,
                                   uint32_t last, uint32_t base) {
  return FIELD(m, 0, 11) | FIELD(n, 12, 23) | FIELD(first, 24, 24) |
         FIELD(last, 25, 25) | FIELD(base, 26, 31);
}

static inline void bb_mxmm_mxfp8(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                                 uint32_t n, uint64_t k, uint32_t first,
                                 uint32_t last, uint32_t base) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) | BB_ITER(k),
      bb_mxmm_cfg(m, n, first, last, base), BB_FUNC7(MXMM_MXFP8));
}

static inline void bb_mxmm_fma32(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                                 uint32_t n, uint64_t k, uint32_t first,
                                 uint32_t last, uint32_t base) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) | BB_ITER(k),
      bb_mxmm_cfg(m, n, first, last, base), BB_FUNC7(MXMM_FMA32));
}

static inline void bb_mxmm_f32(uint32_t a, uint32_t b, uint32_t c, uint32_t m,
                               uint32_t n, uint64_t k, uint32_t first,
                               uint32_t last, uint32_t base) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) | BB_ITER(k),
      bb_mxmm_cfg(m, n, first, last, base), BB_FUNC7(MXMM_F32));
}

static inline void bb_mxmm_mxfp8_window(uint32_t a, uint32_t b, uint32_t c,
                                        uint32_t m, uint32_t n, uint64_t k,
                                        uint32_t first, uint32_t last,
                                        uint32_t base, uint32_t fullK,
                                        uint32_t startK) {
  BUCKYBALL_INSTRUCTION_R_R(BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) |
                                BB_ITER(k),
                            bb_mxmm_cfg(m, n, first, last, base) |
                                FIELD(fullK, 32, 47) | FIELD(startK, 48, 63),
                            BB_FUNC7(MXMM_MXFP8_WINDOW));
}
#endif
