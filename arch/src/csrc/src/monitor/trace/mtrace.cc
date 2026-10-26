#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>
#include <svdpi.h>

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
extern "C" void dpi_mtrace(unsigned int is_write, unsigned int is_shared,
                           unsigned int channel, unsigned int thread_id_lo,
                           unsigned int thread_id_hi, unsigned int rob_id,
                           unsigned int inst_id_lo, unsigned int inst_id_hi,
                           unsigned int vbank_id, unsigned int pbank_id,
                           unsigned int group_id, unsigned int addr,
                           unsigned int write_mask, unsigned int data_lo_lo,
                           unsigned int data_lo_hi, unsigned int data_hi_lo,
                           unsigned int data_hi_hi) {
  if (!bdb_trace_on(BDB_TR_MTRACE)) {
    return;
  }
  init_mtrace();
  const char *scope = svGetNameFromScope(svGetScope());

  if (mtrace_fp) {
    const unsigned long long thread_id =
        ((unsigned long long)thread_id_hi << 32) | thread_id_lo;
    const unsigned long long inst_id =
        ((unsigned long long)inst_id_hi << 32) | inst_id_lo;
    const unsigned long long data_lo =
        ((unsigned long long)data_lo_hi << 32) | data_lo_lo;
    const unsigned long long data_hi =
        ((unsigned long long)data_hi_hi << 32) | data_hi_lo;
    char data_hex[35];
    if (is_write) {
      u128_hex(data_hex, sizeof(data_hex), data_hi, data_lo);
      fprintf(mtrace_fp,
              "{\"type\":\"mtrace\",\"clk\":%llu,\"scope\":\"%s\",\"event\":"
              "\"write\","
              "\"channel\":%u,"
              "\"thread_id\":%llu,\"rob_id\":%u,\"inst_id\":%llu,\"is_shared\":"
              "%u,\"vbank_id\":%u,\"pbank_id\":%u,"
              "\"group_id\":%u,\"addr\":\"0x%08x\",\"write_mask\":%u,\"data\":"
              "\"%s\"}\n",
              (unsigned long long)bdb_rtl_clk, scope, channel, thread_id,
              rob_id, inst_id, is_shared, vbank_id, pbank_id, group_id, addr,
              write_mask, data_hex);
    } else {
      fprintf(mtrace_fp,
              "{\"type\":\"mtrace\",\"clk\":%llu,\"scope\":\"%s\",\"event\":"
              "\"read\",\"channel\":%u,"
              "\"thread_id\":%llu,\"rob_id\":%u,\"inst_id\":%llu,\"is_shared\":"
              "%u,\"vbank_id\":%u,\"pbank_id\":%u,"
              "\"group_id\":%u,\"addr\":\"0x%08x\"}\n",
              (unsigned long long)bdb_rtl_clk, scope, channel, thread_id,
              rob_id, inst_id, is_shared, vbank_id, pbank_id, group_id, addr);
    }
    fflush(mtrace_fp);
  }
}
