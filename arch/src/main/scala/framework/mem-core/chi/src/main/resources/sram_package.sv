package chi_sram_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
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
  `include "sram_test.svh"
endpackage
