`include "core_config.svh"
`ifdef IRQ_PROFILE
`define TEST_IMAGE `IRQ_IMAGE
`include "irq_fixture.svh"
`elsif SUPERVISOR_PROFILE
`define TEST_IMAGE `SUPERVISOR_IMAGE
`include "supervisor_fixture.svh"
`elsif FPU_PROFILE
`define TEST_IMAGE `FPU_IMAGE
`include "fpu_fixture.svh"
`else
`define TEST_IMAGE `CORE_IMAGE
`include "core_fixture.svh"
`endif
package core_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define CF(kind, field) `CORE_``kind``_``field``_OFFSET +: `CORE_``kind``_``field``_WIDTH
  import "DPI-C" function chandle core_ref_create(input string path);
  import "DPI-C" function void core_ref_destroy(input chandle model);
  import "DPI-C" function void core_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function void core_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  `include "core_test.svh"
endpackage
