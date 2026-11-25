package mxquant_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/mxquant_defs.svh"
  `include "common/mxquant_items.svh"
  `include "seq/mxquant_sequences.svh"
  `include "cov/mxquant_cov.svh"
  `include "env/mxquant_scoreboard.svh"
  `include "env/mxquant_cycles.svh"
  `include "env/mxquant_env.svh"
  `include "tests/mxquant_test.svh"
  `include "tests/mxquant_throughput_test.svh"
  `include "tests/mxquant_latency_test.svh"
endpackage
