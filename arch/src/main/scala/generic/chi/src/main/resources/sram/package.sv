package chi_sram_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  import observed_types::*;
  `include "uvm_macros.svh"
  `include "flit_types.svh"
  import "DPI-C" function chandle chi_ref_create(input int unsigned lines);
  import "DPI-C" function void chi_ref_destroy(input chandle model);
  import "DPI-C" function byte unsigned chi_ref_write(
    input chandle model,
    input longint unsigned addr,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  import "DPI-C" function byte unsigned chi_ref_read(
    input chandle model,
    input longint unsigned addr,
    output bit [511:0] data
  );
  `include "agent/item.svh"
  `include "env/model.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/endpoint.svh"
  `include "agent/agent.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
