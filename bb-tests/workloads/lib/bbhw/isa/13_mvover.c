#ifndef _BB_MVOVER_H_
#define _BB_MVOVER_H_

#include "isa.h"

enum { BB_MVOVER_FUNC7 = 0x0d };

/* Move contiguous rows from a private Bank of one Thread to another in the same
 * Core. Bank IDs are Thread-local virtual IDs; row addresses are not byte
 * addresses.
 */
static inline void bb_mvover(uint32_t source_thread, uint32_t source_bank,
                             uint32_t source_row, uint32_t target_thread,
                             uint32_t target_bank, uint32_t target_row,
                             uint32_t rows) {
  if (source_thread > 255 || target_thread > 255 || source_bank > 1023 ||
      target_bank > 1023 || source_row > 65535 || target_row > 65535 || !rows ||
      rows > 65536)
    __builtin_trap();

  uint64_t rs1 = FIELD(source_thread, 0, 7) | FIELD(target_thread, 8, 15) |
                 FIELD(source_bank, 16, 25) | FIELD(target_bank, 26, 35);
  uint64_t rs2 = FIELD(source_row, 0, 15) | FIELD(target_row, 16, 31) |
                 FIELD(rows - 1, 32, 47);
  BUCKYBALL_INSTRUCTION_R_R(rs1, rs2, BB_MVOVER_FUNC7);
}

#endif // _BB_MVOVER_H_
