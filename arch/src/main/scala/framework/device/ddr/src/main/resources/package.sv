`include "parameters.svh"
`define DF(K, F) `DDR_``K``_``F``_OFFSET +: `DDR_``K``_``F``_WIDTH
package ddr_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle ddr_ref_create();
  import "DPI-C" function void ddr_ref_destroy(input chandle model);
  import "DPI-C" function void ddr_ref_program(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input longint unsigned mask
  );
  typedef longint unsigned u64;
  typedef bit [`DDR_REQ_WIDTH-1:0] req_t;
  `include "agent/item.svh"
  `include "env/env.svh"
  `include "tests/test.svh"
endpackage
