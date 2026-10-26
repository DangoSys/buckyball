package dma_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function int unsigned dma_ref_byte(longint unsigned address);
  `include "env/env.svh"
  `include "tests/test.svh"
endpackage
