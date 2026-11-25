#ifndef BB_MXINT5GEMV_H
#define BB_MXINT5GEMV_H
#include <bb_isa.h>
static inline void bb_mxint5gemv(uint32_t a, uint32_t b, uint32_t c, uint32_t k,
                                 uint32_t a_codes, uint32_t a_exponents,
                                 uint32_t b_base, uint32_t c_base,
                                 uint32_t accumulate, uint32_t finalize,
                                 uint32_t update) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(a) | BB_BANK1(b) | BB_BANK2(c) | BB_ITER(k) |
          (accumulate ? BB_READ_WR_OLD : 0) | (update ? BB_UPDATE_WR : 0),
      FIELD(a_codes, 0, 13) | FIELD(a_exponents, 14, 27) |
          FIELD(b_base, 28, 41) | FIELD(c_base, 42, 55) |
          FIELD(accumulate, 56, 56) | FIELD(finalize, 57, 57),
      FUNCT7(mxint5gemv, MXINT5GEMV));
}
#endif
