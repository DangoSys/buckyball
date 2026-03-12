#ifndef _BDB_COUNTER_H_
#define _BDB_COUNTER_H_

#include "isa.h"

#define BDB_COUNTER_FUNC7 48

// subcmd values
#define BDB_CTR_START 0
#define BDB_CTR_STOP 1
#define BDB_CTR_READ 2

// rs2 layout: [3:0]=subcmd, [7:4]=ctr_id, [63:8]=payload
#define BDB_CTR_RS2(subcmd, ctr_id, payload)                                   \
  (((unsigned long long)(payload) << 8) | (((ctr_id)&0xF) << 4) |             \
   ((subcmd)&0xF))

// Start counter ctr_id with user tag
#define bdb_counter_start(ctr_id, tag)                                         \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_START, ctr_id, tag),       \
                            BDB_COUNTER_FUNC7)

// Stop counter ctr_id, output elapsed to trace
#define bdb_counter_stop(ctr_id)                                               \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_STOP, ctr_id, 0),         \
                            BDB_COUNTER_FUNC7)

// Read counter ctr_id current value (non-destructive), output to trace
#define bdb_counter_read(ctr_id)                                               \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_READ, ctr_id, 0),         \
                            BDB_COUNTER_FUNC7)

#endif // _BDB_COUNTER_H_
