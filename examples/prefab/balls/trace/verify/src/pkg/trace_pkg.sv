package trace_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/trace_defs.svh"
  `include "common/trace_items.svh"
  `include "seq/trace_sequences.svh"
  `include "cov/trace_cov.svh"
  `include "env/trace_scoreboard.svh"
  `include "env/trace_cycles.svh"
  `include "env/trace_env.svh"
  `include "tests/trace_test.svh"
  `include "tests/trace_throughput_test.svh"
  `include "tests/trace_latency_test.svh"
endpackage
