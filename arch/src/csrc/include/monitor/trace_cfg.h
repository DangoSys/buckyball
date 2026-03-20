#ifndef MONITOR_TRACE_CFG_H_
#define MONITOR_TRACE_CFG_H_

#include <stdint.h>

enum {
  BDB_TR_ITRACE = 1u << 0,
  BDB_TR_MTRACE = 1u << 1,
  BDB_TR_PMCTRACE = 1u << 2,
  BDB_TR_CTRACE = 1u << 3,
  BDB_TR_BANKTRACE = 1u << 4,
  BDB_TR_ALL = BDB_TR_ITRACE | BDB_TR_MTRACE | BDB_TR_PMCTRACE | BDB_TR_CTRACE |
               BDB_TR_BANKTRACE
};

extern uint32_t bdb_trace_mask;

static inline int bdb_trace_on(uint32_t bit) {
  return (bdb_trace_mask & bit) != 0;
}

#endif
