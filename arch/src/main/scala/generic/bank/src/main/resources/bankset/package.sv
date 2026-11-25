`include "parameters.svh"
package bankset_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  import axis_vip_pkg::*;
  `include "uvm_macros.svh"
  typedef axis_item#(32) item;

  import "DPI-C" function chandle bank_ref_create(input int unsigned entries);
  import "DPI-C" function void bank_ref_destroy(input chandle model);
  import "DPI-C" function int unsigned bank_ref_access(
    input chandle model,
    input int unsigned addr,
    input byte unsigned write,
    input int unsigned data,
    input byte unsigned mask
  );
  `include "agent/item.svh"
  `include "agent/command.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/agent.svh"
  `include "env/coverage.svh"
  `include "env/ref_model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
