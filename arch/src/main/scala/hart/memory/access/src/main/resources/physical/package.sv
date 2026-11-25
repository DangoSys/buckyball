`include "physical_parameters.svh"
package mem_access_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  `define CPUF(kind, field) `CPU_``kind``_``field``_OFFSET +: `CPU_``kind``_``field``_WIDTH
  typedef bit [`ACCESS_REQ_WIDTH-1:0] req_t;
  import "DPI-C" function void mem_access_ref_prepare(
    input longint unsigned addr,
    input int unsigned size,
    write,
    input longint unsigned data,
    input int unsigned atomic_op,
    cacheable,
    normal,
    address_bits,
    output int unsigned target,
    misaligned,
    access_fault,
    output longint unsigned bus_addr,
    bus_data,
    output int unsigned bus_mask,
    atomic_word
  );
  import "DPI-C" function longint unsigned mem_access_ref_result(
    input longint unsigned addr,
    input int unsigned size,
    write,
    signed_load,
    atomic_op,
    cacheable,
    input longint unsigned raw,
    input int unsigned error
  );
  class result_item extends uvm_sequence_item;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(packet, UVM_ALL_ON)
    `uvm_object_utils_end
    logic [`ACCESS_RESP_WIDTH-1:0] packet;
    function new(string name = "result");
      super.new(name);
    endfunction
  endclass
  `include "env/env.svh"
  `include "tests/test.svh"
endpackage
