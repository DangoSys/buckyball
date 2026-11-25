package mxint8quant_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/mxint8quant_defs.svh"
  `include "common/mxint8quant_items.svh"
  `include "seq/mxint8quant_sequences.svh"
  `include "cov/mxint8quant_cov.svh"
  `include "env/mxint8quant_cycles.svh"
  `include "env/mxint8quant_scoreboard.svh"
  `include "env/mxint8quant_env.svh"
  `include "tests/mxint8quant_test.svh"
  `include "tests/mxint8quant_throughput_test.svh"
  `include "tests/mxint8quant_latency_test.svh"
endpackage
