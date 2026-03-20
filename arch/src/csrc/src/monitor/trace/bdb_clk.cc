#include "monitor/trace_cfg.h"

// Harness reference clock cycle index (updated from RTL via dpi_bdb_set_clk
// each posedge).
uint64_t bdb_rtl_clk = 0;

extern "C" void dpi_bdb_set_clk(unsigned long long c) {
  bdb_rtl_clk = (uint64_t)c;
}
