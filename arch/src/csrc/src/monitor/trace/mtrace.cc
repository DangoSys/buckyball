#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>

// Global log file pointer (shared with monitor.cc)
extern const char *log_path;
static FILE *mtrace_fp = NULL;

// Initialize mtrace logging
static void init_mtrace() {
  if (mtrace_fp == NULL && log_path != NULL) {
    mtrace_fp = fopen(log_path, "a");
    if (mtrace_fp == NULL) {
      panic("Failed to open mtrace log file: %s", log_path);
    }
  }
}

static void u128_hex(char *buf, size_t n, unsigned long long hi,
                     unsigned long long lo) {
  int ret = snprintf(buf, n, "0x%016llx%016llx", hi, lo);
  if (ret < 0 || (size_t)ret >= n) {
    panic("snprintf failed in mtrace u128_hex");
  }
}

// DPI-C function for memory trace (mtrace)
// Called when MemBackend performs read/write operations
extern "C" void dpi_mtrace(unsigned char is_write, // 1 = write, 0 = read
                           unsigned char is_shared, unsigned int channel,
                           unsigned long long hart_id, unsigned int vbank_id,
                           unsigned int pbank_id, unsigned int group_id,
                           unsigned int addr, unsigned long long data_lo,
                           unsigned long long data_hi) {
  if (!bdb_trace_on(BDB_TR_MTRACE)) {
    return;
  }
  init_mtrace();

  if (mtrace_fp) {
    char data_hex[35];
    if (is_write) {
      u128_hex(data_hex, sizeof(data_hex), data_hi, data_lo);
      fprintf(
          mtrace_fp,
          "{\"type\":\"mtrace\",\"clk\":%llu,\"event\":\"write\","
          "\"channel\":%u,"
          "\"hart_id\":%llu,\"is_shared\":%u,\"vbank_id\":%u,\"pbank_id\":%u,"
          "\"group_id\":%u,\"addr\":\"0x%08x\",\"data\":\"%s\"}\n",
          (unsigned long long)bdb_rtl_clk, channel, hart_id, is_shared,
          vbank_id, pbank_id, group_id, addr, data_hex);
    } else {
      fprintf(
          mtrace_fp,
          "{\"type\":\"mtrace\",\"clk\":%llu,\"event\":\"read\",\"channel\":%u,"
          "\"hart_id\":%llu,\"is_shared\":%u,\"vbank_id\":%u,\"pbank_id\":%u,"
          "\"group_id\":%u,\"addr\":\"0x%08x\"}\n",
          (unsigned long long)bdb_rtl_clk, channel, hart_id, is_shared,
          vbank_id, pbank_id, group_id, addr);
    }
    fflush(mtrace_fp);
  }
}
