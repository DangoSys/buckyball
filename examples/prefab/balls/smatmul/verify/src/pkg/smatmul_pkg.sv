package smatmul_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/smatmul_defs.svh"
  `include "common/smatmul_items.svh"
  `include "seq/smatmul_sequences.svh"
  `include "cov/smatmul_cov.svh"
  `include "env/smatmul_scoreboard.svh"
  `include "env/smatmul_cycles.svh"
  `include "env/smatmul_env.svh"
  `include "tests/smatmul_test.svh"
  `include "tests/smatmul_throughput_test.svh"
  `include "tests/smatmul_latency_test.svh"
  `include "tests/smatmul_m24_test.svh"
endpackage
