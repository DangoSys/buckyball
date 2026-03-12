#ifndef _BDB_BACKDOOR_H_
#define _BDB_BACKDOOR_H_

#include "isa.h"

#define BDB_BACKDOOR_FUNC7 49

// Backdoor write: DPI-C provides data, write to external bank
// bank_id = target bank, iter = number of rows
#define bdb_backdoor_write(bank_id, iter)                                      \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK2(bank_id) | BB_WR | BB_ITER(iter)), 0, BDB_BACKDOOR_FUNC7)

// Backdoor read: read from external bank, output via DPI-C
// bank_id = source bank, iter = number of rows
#define bdb_backdoor_read(bank_id, iter)                                       \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_RD0 | BB_ITER(iter)), 0, BDB_BACKDOOR_FUNC7)

// Backdoor peek: read single row from external bank
#define bdb_backdoor_peek(bank_id, row_count)                                  \
  BUCKYBALL_INSTRUCTION_R_R(                                                   \
      (BB_BANK0(bank_id) | BB_RD0 | BB_ITER(row_count)), 0,                   \
      BDB_BACKDOOR_FUNC7)

#endif // _BDB_BACKDOOR_H_
