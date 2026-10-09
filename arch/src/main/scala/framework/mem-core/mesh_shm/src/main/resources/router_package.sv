package router_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef virtual axis_if #(128, 8, 4, 45) router_vif;
  import "DPI-C" function int unsigned mesh_route(
    input int unsigned destination,
    input int unsigned row,
    input int unsigned col
  );
  `include "router_item.svh"
  `include "router_test.svh"
endpackage
