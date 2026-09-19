package arbiter_pkg;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function chandle axis_arbiter_create();
  import "DPI-C" function void axis_arbiter_destroy(input chandle model);
  import "DPI-C" function int unsigned axis_arbiter_step(
    input chandle model,
    input int unsigned valid,
    input byte unsigned ready,
    input int unsigned last
  );

  `include "arbiter_driver.svh"
  `include "arbiter_checker.svh"
  `include "arbiter_test.svh"
endpackage
