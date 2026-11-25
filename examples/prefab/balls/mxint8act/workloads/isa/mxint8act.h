#ifndef _BB_MXINT8ACT_H_
#define _BB_MXINT8ACT_H_

#include <bb_isa.h>

/* Integer activation of `rows` packed MXINT8 rows of k codes (row r at bank
 * row in_base + r * rows-per-row of `input`; when gated, the gate rows follow
 * the activation rows) through the 4096-entry Q26 table at bank row table_base
 * of `table`, into `output` from bank row out_base. */
#define bb_mxint8act(input_bank, table_bank, output_bank, rows, k, in_base,    \
                     out_base, table_base, gated)                              \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      BB_BANK0(input_bank) | BB_BANK1(table_bank) | BB_BANK2(output_bank) |    \
          BB_ITER(rows),                                                       \
      (uint64_t)(gated) | ((uint64_t)(k) << 1) | ((uint64_t)(in_base) << 17) | \
          ((uint64_t)(out_base) << 31) | ((uint64_t)(table_base) << 45),       \
      FUNCT7(mxint8act, MXINT8ACT))

#endif // _BB_MXINT8ACT_H_
