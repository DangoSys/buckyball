`include "lsu_config.svh"
package lsu_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define LF(kind, field) `CORE_``kind``_``field``_OFFSET +: `CORE_``kind``_``field``_WIDTH
  `include "lsu_test.svh"
endpackage
