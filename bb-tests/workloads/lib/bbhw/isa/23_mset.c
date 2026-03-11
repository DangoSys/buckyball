#ifndef _BB_MSET_H_
#define _BB_MSET_H_

#include "isa.h"

#define BB_MSET_FUNC7 23

// Shared-bank software IDs are 1-based: shared_id=1 maps to hardware vbank_id=32.
#define BB_SHARED_BANK_OFFSET 31UL
#define BB_SHARED_BANK_ID(shared_id) ((shared_id) + BB_SHARED_BANK_OFFSET)

#define bb_mset(bank_id, alloc, row, col)                           \
  BUCKYBALL_INSTRUCTION_R_R(                                        \
      (BB_BANK0(bank_id) | BB_WR),                                  \
      (FIELD(row, 0, 4) | FIELD(col, 5, 9) | FIELD(alloc, 10, 10)), \
      BB_MSET_FUNC7)

#define bb_mem_release(bank_id) bb_mset(bank_id, 0, 0, 0);

#define bb_mem_alloc(bank_id, row, col) bb_mset(bank_id, 1, row, col)

#define bb_mem_alloc_private(bank_id, row, col) bb_mem_alloc(bank_id, row, col)
#define bb_mem_alloc_shared(shared_id, row, col) \
  bb_mem_alloc(BB_SHARED_BANK_ID(shared_id), row, col)

#define bb_mem_release_private(bank_id) bb_mem_release(bank_id)
#define bb_mem_release_shared(shared_id) bb_mem_release(BB_SHARED_BANK_ID(shared_id))

#endif // _BB_MSET_H_
