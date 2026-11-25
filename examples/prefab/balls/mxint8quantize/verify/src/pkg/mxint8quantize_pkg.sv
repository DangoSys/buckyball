package mxint8quantize_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/mxint8quantize_defs.svh"
  `include "common/mxint8quantize_items.svh"
  `include "seq/mxint8quantize_sequences.svh"
  `include "cov/mxint8quantize_cov.svh"
  `include "env/mxint8quantize_scoreboard.svh"
  `include "env/mxint8quantize_cycles.svh"
  `include "env/mxint8quantize_env.svh"
  `include "tests/mxint8quantize_test.svh"
  `include "tests/mxint8quantize_throughput_test.svh"
  `include "tests/mxint8quantize_latency_test.svh"
endpackage
