#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Global log file pointer (shared with monitor.cc)
extern const char *log_path;
static FILE *ctrace_fp = NULL;

static void init_ctrace() {
  if (ctrace_fp == NULL && log_path != NULL) {
    ctrace_fp = fopen(log_path, "a");
    if (ctrace_fp == NULL) {
      panic("Failed to open ctrace log file: %s", log_path);
    }
  }
}

// ================================================================
// Cycle counter DPI-C
// ================================================================

extern "C" void dpi_ctrace(unsigned int subcmd, unsigned int ctr_id,
                           unsigned int tag_lo, unsigned int tag_hi,
                           unsigned int elapsed_lo, unsigned int elapsed_hi,
                           unsigned int cycle_lo, unsigned int cycle_hi) {
  if (!bdb_trace_on(BDB_TR_CTRACE)) {
    return;
  }
  init_ctrace();
  unsigned long long tag = ((unsigned long long)tag_hi << 32) | tag_lo;
  unsigned long long elapsed =
      ((unsigned long long)elapsed_hi << 32) | elapsed_lo;
  unsigned long long cycle = ((unsigned long long)cycle_hi << 32) | cycle_lo;

  if (ctrace_fp) {
    switch (subcmd) {
    case 0: // CTR_START
      fprintf(ctrace_fp,
              "{\"type\":\"ctrace\",\"clk\":%llu,\"event\":\"ctr_start\",\"ctr_"
              "id\":%u,"
              "\"tag\":\"0x%llX\",\"cycle\":%llu}\n",
              (unsigned long long)bdb_rtl_clk, ctr_id, tag, cycle);
      break;
    case 1: // CTR_STOP
      fprintf(ctrace_fp,
              "{\"type\":\"ctrace\",\"clk\":%llu,\"event\":\"ctr_stop\",\"ctr_"
              "id\":%u,"
              "\"tag\":\"0x%llX\",\"elapsed\":%llu,\"cycle\":%llu}\n",
              (unsigned long long)bdb_rtl_clk, ctr_id, tag, elapsed, cycle);
      break;
    case 2: // CTR_READ
      fprintf(ctrace_fp,
              "{\"type\":\"ctrace\",\"clk\":%llu,\"event\":\"ctr_read\",\"ctr_"
              "id\":%u,"
              "\"current\":%llu,\"cycle\":%llu}\n",
              (unsigned long long)bdb_rtl_clk, ctr_id, elapsed, cycle);
      break;
    }
    fflush(ctrace_fp);
  }
}

// RTL calls this when a Ball instruction writes a bank: logs the bank hash
extern "C" void dpi_btrace(unsigned int inst_id_lo, unsigned int inst_id_hi,
                           unsigned int thread_id_lo, unsigned int thread_id_hi,
                           unsigned int owner_thread_id_lo,
                           unsigned int owner_thread_id_hi,
                           unsigned int w0_vbank, unsigned int w0_hash) {
  if (!bdb_trace_on(BDB_TR_BANKTRACE)) {
    return;
  }
  init_ctrace();
  if (ctrace_fp) {
    fprintf(ctrace_fp,
            "{\"type\":\"banktrace\",\"clk\":%llu,\"event\":\"btrace\","
            "\"inst_id\":%llu,\"thread_id\":%llu,\"owner_thread_id\":%llu,"
            "\"vbank\":%u,\"hash\":\"0x%08X\"}\n",
            (unsigned long long)bdb_rtl_clk,
            ((unsigned long long)inst_id_hi << 32) | inst_id_lo,
            ((unsigned long long)thread_id_hi << 32) | thread_id_lo,
            ((unsigned long long)owner_thread_id_hi << 32) | owner_thread_id_lo,
            w0_vbank, w0_hash);
    fflush(ctrace_fp);
  }
}
