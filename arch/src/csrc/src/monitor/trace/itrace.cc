#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>
#include <svdpi.h>

// Global log file pointer (shared with monitor.cc)
extern const char *log_path;
static FILE *itrace_fp = NULL;

// Initialize itrace logging
static void init_itrace() {
  if (itrace_fp == NULL && log_path != NULL) {
    itrace_fp = fopen(log_path, "a");
    if (itrace_fp == NULL) {
      panic("Failed to open itrace log file: %s", log_path);
    }
  }
}

static void u64_hex(char *buf, size_t n, unsigned long long v) {
  int ret = snprintf(buf, n, "0x%016llx", v);
  if (ret < 0 || (size_t)ret >= n) {
    panic("snprintf failed in itrace u64_hex");
  }
}

// DPI-C function for instruction trace (itrace)
// Called when an instruction is allocated/issued/completed in GlobalROB
extern "C" void
dpi_itrace(unsigned int is_issue, // 2 = alloc, 1 = issue, 0 = complete
           unsigned int thread_id_lo, unsigned int thread_id_hi,
           unsigned int rob_id, unsigned int inst_id_lo,
           unsigned int inst_id_hi, unsigned int domain_id, unsigned int funct,
           unsigned int pc_lo, unsigned int pc_hi, unsigned int rs1_lo,
           unsigned int rs1_hi, unsigned int rs2_lo, unsigned int rs2_hi) {
  if (!bdb_trace_on(BDB_TR_ITRACE)) {
    return;
  }
  init_itrace();
  const char *scope = svGetNameFromScope(svGetScope());

  if (itrace_fp) {
    unsigned long long thread_id =
        ((unsigned long long)thread_id_hi << 32) | thread_id_lo;
    unsigned long long inst_id =
        ((unsigned long long)inst_id_hi << 32) | inst_id_lo;
    unsigned long long pc = ((unsigned long long)pc_hi << 32) | pc_lo;
    unsigned long long rs1 = ((unsigned long long)rs1_hi << 32) | rs1_lo;
    unsigned long long rs2 = ((unsigned long long)rs2_hi << 32) | rs2_lo;
    char pc_hex[19];
    char rs1_hex[19];
    char rs2_hex[19];
    u64_hex(pc_hex, sizeof(pc_hex), pc);
    u64_hex(rs1_hex, sizeof(rs1_hex), rs1);
    u64_hex(rs2_hex, sizeof(rs2_hex), rs2);
    if (is_issue == 2) {
      fprintf(itrace_fp,
              "{\"type\":\"itrace\",\"clk\":%llu,\"scope\":\"%s\",\"event\":"
              "\"alloc\",\"thread_"
              "id\":%llu,"
              "\"rob_id\":%u,\"inst_id\":%llu,\"domain_id\":%u,\"funct\":\"0x%"
              "02x\","
              "\"pc\":\"%s\",\"rs1\":\"%s\",\"rs2\":\"%s\"}\n",
              (unsigned long long)bdb_rtl_clk, scope, thread_id, rob_id,
              inst_id, domain_id, funct, pc_hex, rs1_hex, rs2_hex);
    } else if (is_issue == 1) {
      fprintf(itrace_fp,
              "{\"type\":\"itrace\",\"clk\":%llu,\"scope\":\"%s\",\"event\":"
              "\"issue\",\"thread_"
              "id\":%llu,"
              "\"rob_id\":%u,\"inst_id\":%llu,\"domain_id\":%u,\"funct\":\"0x%"
              "02x\","
              "\"pc\":\"%s\",\"rs1\":\"%s\",\"rs2\":\"%s\"}\n",
              (unsigned long long)bdb_rtl_clk, scope, thread_id, rob_id,
              inst_id, domain_id, funct, pc_hex, rs1_hex, rs2_hex);
    } else {
      fprintf(
          itrace_fp,
          "{\"type\":\"itrace\",\"clk\":%llu,\"scope\":\"%s\",\"event\":"
          "\"complete\","
          "\"thread_id\":%llu,\"rob_id\":%u,\"inst_id\":%llu,\"domain_id\":%u,"
          "\"funct\":\"0x%02x\",\"pc\":\"%s\"}\n",
          (unsigned long long)bdb_rtl_clk, scope, thread_id, rob_id, inst_id,
          domain_id, funct, pc_hex);
    }
    fflush(itrace_fp);
  }
}

extern "C" void dpi_ant_trace(unsigned int thread_lo, unsigned int thread_hi,
                              unsigned int pc_lo, unsigned int pc_hi,
                              unsigned int instruction, unsigned int rd,
                              unsigned int data_lo, unsigned int data_hi) {
  if (!bdb_trace_on(BDB_TR_ITRACE))
    return;
  init_itrace();
  const char *scope = svGetNameFromScope(svGetScope());
  if (!itrace_fp)
    return;
  const auto thread =
      (static_cast<unsigned long long>(thread_hi) << 32) | thread_lo;
  const auto pc = (static_cast<unsigned long long>(pc_hi) << 32) | pc_lo;
  const auto data = (static_cast<unsigned long long>(data_hi) << 32) | data_lo;
  fprintf(itrace_fp,
          "{\"type\":\"ant_trace\",\"clk\":%llu,\"scope\":\"%s\",\"thread_id\":"
          "%llu,"
          "\"pc\":\"0x%016llx\",\"instruction\":\"0x%08x\","
          "\"rd\":%u,\"data\":\"0x%016llx\"}\n",
          static_cast<unsigned long long>(bdb_rtl_clk), scope, thread, pc,
          instruction, rd, data);
  fflush(itrace_fp);
}
