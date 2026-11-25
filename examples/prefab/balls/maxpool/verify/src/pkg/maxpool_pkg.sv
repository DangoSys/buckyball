package maxpool_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/maxpool_defs.svh"
  `include "common/maxpool_items.svh"
  `include "seq/maxpool_sequences.svh"
  `include "cov/maxpool_cov.svh"
  `include "env/maxpool_scoreboard.svh"
  `include "env/maxpool_cycles.svh"
  `include "env/maxpool_env.svh"
  `include "tests/maxpool_test.svh"
  `include "tests/maxpool_throughput_test.svh"
  `include "tests/maxpool_latency_test.svh"
endpackage
