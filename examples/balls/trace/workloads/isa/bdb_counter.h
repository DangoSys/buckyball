#ifndef _BDB_COUNTER_H_
#define _BDB_COUNTER_H_

#include <bbhw/isa/bb_func7.h>
#include <bbhw/isa/isa.h>

// subcmd values
#define BDB_CTR_START 0
#define BDB_CTR_STOP 1
#define BDB_CTR_READ 2

// rs2 layout: [3:0]=subcmd, [7:4]=ctr_id, [63:8]=payload
#define BDB_CTR_RS2(subcmd, ctr_id, payload)                                   \
  (((unsigned long long)(payload) << 8) | (((ctr_id) & 0xF) << 4) |            \
   ((subcmd) & 0xF))

// Start counter ctr_id with user tag
#define bdb_counter_start(ctr_id, tag)                                         \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_START, ctr_id, tag),        \
                            BB_FUNC7(BDB_COUNTER))

// Stop counter ctr_id, output elapsed to trace
#define bdb_counter_stop(ctr_id)                                               \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_STOP, ctr_id, 0),           \
                            BB_FUNC7(BDB_COUNTER))

// Read counter ctr_id current value (non-destructive), output to trace
#define bdb_counter_read(ctr_id)                                               \
  BUCKYBALL_INSTRUCTION_R_R(0, BDB_CTR_RS2(BDB_CTR_READ, ctr_id, 0),           \
                            BB_FUNC7(BDB_COUNTER))

#endif // _BDB_COUNTER_H_
