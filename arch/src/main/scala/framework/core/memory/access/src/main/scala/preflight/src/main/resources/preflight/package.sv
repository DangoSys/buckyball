`include "parameters.svh"
`define PF(K, F) `PF_``K``_``F``_OFFSET +: `PF_``K``_``F``_WIDTH
package preflight_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  typedef longint unsigned u64;
  typedef bit [`PF_CMD_WIDTH-1:0] command_t;
  import "DPI-C" function chandle preflight_ref_create(
    input int unsigned address_bits,
    beat
  );
  import "DPI-C" function void preflight_ref_destroy(input chandle model);
  import "DPI-C" function void preflight_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
  );
  import "DPI-C" function void preflight_ref_policy(
    input chandle model,
    input int unsigned pte_valid,
    input longint unsigned pte,
    input int unsigned range_valid,
    input longint unsigned first,
    last
  );
  import "DPI-C" function int unsigned preflight_ref_submit(
    input chandle model,
    input int unsigned id,
    input longint unsigned base,
    input int unsigned rows,
    cols,
    span,
    col_stride,
    row_stride,
    write,
    mode,
    input longint unsigned root,
    input int unsigned privilege,
    sum,
    mxr
  );
  import "DPI-C" function int unsigned preflight_ref_count(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_expected(
    input chandle model,
    input int unsigned id,
    index,
    output longint unsigned va,
    pa,
    output int unsigned bytes,
    write,
    last,
    error
  );
  import "DPI-C" function int unsigned preflight_ref_authorize(
    input chandle model,
    input int unsigned id,
    input longint unsigned pa,
    input int unsigned bytes,
    write,
    pte,
    privilege,
    output int unsigned allow
  );
  import "DPI-C" function int unsigned preflight_ref_pte(
    input chandle model,
    input int unsigned id,
    input longint unsigned address,
    output longint unsigned value,
    output int unsigned error
  );
  import "DPI-C" function int unsigned preflight_ref_pending(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_retire(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_reset(input chandle model);
  class result_item extends uvm_sequence_item;
    bit [`PF_OUT_WIDTH-1:0] packet;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(packet, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  `include "env/env.svh"
  `include "tests/test.svh"
endpackage
