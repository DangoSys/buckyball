package arbiter_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function chandle axis_arbiter_create();
  import "DPI-C" function void axis_arbiter_destroy(input chandle model);
  import "DPI-C" function int unsigned axis_arbiter_step(
    input chandle model,
    input int unsigned valid,
    input byte unsigned ready,
    input int unsigned last
  );

  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/agent.svh"
  `include "env/ref_model.svh"
  `include "env/coverage.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
