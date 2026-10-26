#ifndef _BB_INT8ADD_H_
#define _BB_INT8ADD_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>
#include <stdint.h>

static inline uint64_t bb_int8add_rs2(float lhs_ratio, float rhs_ratio) {
  union {
    float f;
    uint32_t u;
  } lhs = {.f = lhs_ratio}, rhs = {.f = rhs_ratio};
  return (uint64_t)lhs.u | ((uint64_t)rhs.u << 32);
}

#define bb_int8add(lhs_bank, rhs_bank, output_bank, iter, lhs_ratio,           \
                   rhs_ratio)                                                  \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(lhs_bank) | BB_BANK1(rhs_bank) |         \
                             BB_BANK2(output_bank) | BB_ITER(iter)),           \
                            bb_int8add_rs2(lhs_ratio, rhs_ratio),              \
                            BB_FUNC7(INT8ADD))

#define bb_int8add_relu(lhs_bank, rhs_bank, output_bank, iter, lhs_ratio,      \
                        rhs_ratio)                                             \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(lhs_bank) | BB_BANK1(rhs_bank) |         \
                             BB_BANK2(output_bank) | BB_ITER(iter)),           \
                            bb_int8add_rs2(lhs_ratio, rhs_ratio),              \
                            BB_FUNC7(INT8ADD_RELU))

#endif // _BB_INT8ADD_H_
