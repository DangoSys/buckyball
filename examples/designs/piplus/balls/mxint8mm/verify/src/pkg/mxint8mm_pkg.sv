package mxint8mm_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/mxint8mm_defs.svh"
  `include "common/mxint8mm_items.svh"
  `include "seq/mxint8mm_sequences.svh"
  `include "cov/mxint8mm_cov.svh"
  `include "env/mxint8mm_cycles.svh"
  `include "env/mxint8mm_scoreboard.svh"
  `include "env/mxint8mm_env.svh"
  `include "tests/mxint8mm_test.svh"
  `include "tests/mxint8mm_throughput_test.svh"
  `include "tests/mxint8mm_latency_test.svh"
endpackage
