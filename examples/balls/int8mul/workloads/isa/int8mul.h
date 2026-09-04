#ifndef _BB_INT8MUL_H_
#define _BB_INT8MUL_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>
#include <stdint.h>

static inline uint64_t bb_int8mul_rs2(float ratio, uint32_t gate_row) {
  union {
    float f;
    uint32_t u;
  } value = {.f = ratio};
  return (uint64_t)value.u | ((uint64_t)gate_row << 32);
}

#define bb_int8mul(gate_bank, input_bank, output_bank, iter, ratio, gate_row)  \
  BUCKYBALL_INSTRUCTION_R_R((BB_BANK0(gate_bank) | BB_BANK1(input_bank) |      \
                             BB_BANK2(output_bank) | BB_ITER(iter)),           \
                            bb_int8mul_rs2(ratio, gate_row),                   \
                            BB_FUNC7(INT8MUL))

#endif // _BB_INT8MUL_H_
