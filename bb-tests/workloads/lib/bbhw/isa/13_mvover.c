#ifndef _BB_MVOVER_H_
#define _BB_MVOVER_H_

#include "isa.h"

enum { BB_MVOVER_FUNC7 = 0x0d };

/* Move contiguous rows from a private Bank of one Core to another in the same
 * Tile. Bank IDs are Core-local virtual IDs; row addresses are not byte
 * addresses.
 */
static inline void bb_mvover(uint32_t source_core, uint32_t source_bank,
                             uint32_t source_row, uint32_t target_core,
                             uint32_t target_bank, uint32_t target_row,
                             uint32_t rows) {
  if (source_core > 255 || target_core > 255 || source_bank > 1023 ||
      target_bank > 1023 || source_row > 65535 || target_row > 65535 || !rows ||
      rows > 65536)
    __builtin_trap();

  uint64_t rs1 = FIELD(source_core, 0, 7) | FIELD(target_core, 8, 15) |
                 FIELD(source_bank, 16, 25) | FIELD(target_bank, 26, 35);
  uint64_t rs2 = FIELD(source_row, 0, 15) | FIELD(target_row, 16, 31) |
                 FIELD(rows - 1, 32, 47);
  BUCKYBALL_INSTRUCTION_R_R(rs1, rs2, BB_MVOVER_FUNC7);
}

#endif // _BB_MVOVER_H_
