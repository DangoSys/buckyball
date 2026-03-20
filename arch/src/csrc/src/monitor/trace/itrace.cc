#include "monitor/trace.h"
#include "monitor/trace_cfg.h"
#include "utils/debug.h"
#include <stdint.h>
#include <stdio.h>

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

// bank_enable encoding (funct7[6:4]):
//   000 = none, 001 = 1rd, 010 = 1wr, 011 = 1rd+1wr, 100 = 2rd+1wr
//   101/110/111 = none (extended opcode space)
static const char *bank_enable_str(unsigned char enable) {
  switch (enable) {
  case 0:
    return "---";
  case 1:
    return "R--";
  case 2:
    return "--W";
  case 3:
    return "R-W";
  case 4:
    return "RRW";
  default:
    return "---"; // 5,6,7 = no bank access (extended)
  }
}

static void u64_hex(char *buf, size_t n, unsigned long long v) {
  int ret = snprintf(buf, n, "0x%016llx", v);
  if (ret < 0 || (size_t)ret >= n) {
    panic("snprintf failed in itrace u64_hex");
  }
}

// DPI-C function for instruction trace (itrace)
// Called when an instruction is issued or completed in GlobalROB
extern "C" void dpi_itrace(unsigned char is_issue, // 1 = issue, 0 = complete
                           unsigned int rob_id, unsigned int domain_id,
                           unsigned int funct, unsigned long long rs1,
                           unsigned long long rs2, unsigned char bank_enable) {
  if (!bdb_trace_on(BDB_TR_ITRACE)) {
    return;
  }
  init_itrace();

  if (itrace_fp) {
    char rs1_hex[19];
    char rs2_hex[19];
    u64_hex(rs1_hex, sizeof(rs1_hex), rs1);
    u64_hex(rs2_hex, sizeof(rs2_hex), rs2);
    if (is_issue) {
      fprintf(itrace_fp,
              "{\"type\":\"itrace\",\"event\":\"issue\",\"rob_id\":%u,"
              "\"domain_id\":%u,\"funct\":\"0x%02x\",\"bank_enable\":%u,"
              "\"bank\":\"%s\",\"rs1\":\"%s\",\"rs2\":\"%s\"}\n",
              rob_id, domain_id, funct, bank_enable,
              bank_enable_str(bank_enable), rs1_hex, rs2_hex);
    } else {
      fprintf(itrace_fp,
              "{\"type\":\"itrace\",\"event\":\"complete\",\"rob_id\":%u,"
              "\"domain_id\":%u,\"funct\":\"0x%02x\",\"bank_enable\":%u,"
              "\"bank\":\"%s\"}\n",
              rob_id, domain_id, funct, bank_enable,
              bank_enable_str(bank_enable));
    }
    fflush(itrace_fp);
  }
}
