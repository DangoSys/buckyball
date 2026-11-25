#ifndef BB_INTSOFTMAX_H
#define BB_INTSOFTMAX_H
#include <bb_isa.h>
static inline void bb_intsoftmax(uint32_t input, uint32_t table,
                                 uint32_t output, uint32_t rows,
                                 uint32_t columns) {
  BUCKYBALL_INSTRUCTION_R_R(
      BB_BANK0(input) | BB_BANK1(table) | BB_BANK2(output) | BB_ITER(rows),
      FIELD(columns, 0, 7), FUNCT7(intsoftmax, INTSOFTMAX));
}
#endif
