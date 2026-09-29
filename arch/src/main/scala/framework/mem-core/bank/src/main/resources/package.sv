package bank_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function chandle bank_ref_create(input int unsigned entries);
  import "DPI-C" function void bank_ref_destroy(input chandle model);
  import "DPI-C" function int unsigned bank_ref_access(
    input chandle model,
    input int unsigned addr,
    input byte unsigned write,
    input int unsigned data,
    input byte unsigned mask
  );

  `include "item.svh"
  `include "ref_model.svh"
  `include "env.svh"
  `include "test.svh"
endpackage
