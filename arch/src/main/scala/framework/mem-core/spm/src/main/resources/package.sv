package spm_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle spm_create(
    input longint unsigned base,
    input int unsigned bytes
  );
  import "DPI-C" function void spm_destroy(input chandle model);
  import "DPI-C" function int unsigned spm_access(
    input chandle model,
    input longint unsigned address,
    input int unsigned size,
    input byte unsigned write,
    input int unsigned mask,
    input longint unsigned low,
    high,
    input byte unsigned read_only,
    output longint unsigned out_low,
    out_high
  );
  class response_item extends uvm_sequence_item;
    logic [127:0] data;
    logic error;
    `uvm_object_utils_begin(response_item)
      `uvm_field_int(data, UVM_ALL_ON)
      `uvm_field_int(error, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "response_item");
      super.new(name);
    endfunction
  endclass
endpackage
