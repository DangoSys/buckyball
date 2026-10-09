package shared_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef virtual axis_if #(128, 8, 4, 17) request_vif;
  typedef virtual axis_if #(128, 8, 0, 2) response_vif;
  import "DPI-C" function chandle mesh_banks_create();
  import "DPI-C" function void mesh_banks_destroy(input chandle model);
  import "DPI-C" function int unsigned mesh_bank_access(
    input chandle model,
    input int unsigned bank,
    input int unsigned row,
    input byte unsigned write,
    input bit [127:0] data,
    input int unsigned mask,
    output bit [127:0] result
  );
  `include "shared_item.svh"
  `include "shared_test.svh"
endpackage
