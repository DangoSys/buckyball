`include "fetch_config.svh"
package fetch_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define FF(kind, field) `FETCH_``kind``_``field``_OFFSET +: `FETCH_``kind``_``field``_WIDTH
  import "DPI-C" function longint unsigned fetch_ref_word(
    input longint unsigned address,
    input int unsigned version
  );
  import "DPI-C" function int unsigned fetch_ref_group(
    input longint unsigned address,
    input int unsigned version
  );
  import "DPI-C" function int unsigned fetch_ref_instruction(
    input longint unsigned address,
    input int unsigned version
  );
  `include "test.svh"
endpackage
