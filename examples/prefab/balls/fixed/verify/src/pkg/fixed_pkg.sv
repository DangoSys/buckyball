package fixed_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/fixed_defs.svh"
  `include "common/fixed_items.svh"
  `include "seq/fixed_sequences.svh"
  `include "cov/fixed_cov.svh"
  `include "env/fixed_scoreboard.svh"
  `include "env/fixed_cycles.svh"
  `include "env/fixed_env.svh"
  `include "tests/fixed_test.svh"
  `include "tests/fixed_throughput_test.svh"
  `include "tests/fixed_latency_test.svh"
endpackage
