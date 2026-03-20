#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>

// Global log file pointer (shared with monitor.cc)
extern const char *log_path;
static FILE *pmctrace_fp = NULL;

// Initialize pmctrace logging
static void init_pmctrace() {
  if (pmctrace_fp == NULL && log_path != NULL) {
    pmctrace_fp = fopen(log_path, "a");
    if (pmctrace_fp == NULL) {
      panic("Failed to open pmctrace log file: %s", log_path);
    }
  }
}

// DPI-C function for Ball PMC trace
// Called when a Ball completes a task, reports elapsed cycles
extern "C" void dpi_pmctrace(unsigned int ball_id, unsigned int rob_id,
                             unsigned long long elapsed) {
  if (!bdb_trace_on(BDB_TR_PMCTRACE)) {
    return;
  }
  init_pmctrace();

  if (pmctrace_fp) {
    fprintf(
        pmctrace_fp,
        "{\"type\":\"pmctrace\",\"clk\":%llu,\"event\":\"ball\",\"ball_id\":%u,"
        "\"rob_id\":%u,\"elapsed\":%llu}\n",
        (unsigned long long)bdb_rtl_clk, ball_id, rob_id, elapsed);
    fflush(pmctrace_fp);
  }
}

// DPI-C function for Memory PMC trace
// Called when a load/store completes, reports elapsed cycles
extern "C" void dpi_mem_pmctrace(unsigned char is_store, unsigned int rob_id,
                                 unsigned long long elapsed) {
  if (!bdb_trace_on(BDB_TR_PMCTRACE)) {
    return;
  }
  init_pmctrace();

  if (pmctrace_fp) {
    if (is_store) {
      fprintf(pmctrace_fp,
              "{\"type\":\"pmctrace\",\"clk\":%llu,\"event\":\"store\","
              "\"rob_id\":%u,\"elapsed\":%llu}\n",
              (unsigned long long)bdb_rtl_clk, rob_id, elapsed);
    } else {
      fprintf(pmctrace_fp,
              "{\"type\":\"pmctrace\",\"clk\":%llu,\"event\":\"load\","
              "\"rob_id\":%u,\"elapsed\":%llu}\n",
              (unsigned long long)bdb_rtl_clk, rob_id, elapsed);
    }
    fflush(pmctrace_fp);
  }
}
