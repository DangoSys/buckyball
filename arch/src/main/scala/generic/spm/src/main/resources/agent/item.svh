class request_item extends uvm_sequence_item;
  logic [63:0] address;
  logic [$clog2($clog2(`SPM_BITS/8)+2)-1:0] size;
  logic write;
  logic [`SPM_BITS-1:0] data;
  logic [`SPM_BITS/8-1:0] mask;
  logic read_only;
  int unsigned stall;
  logic cancel;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(address, UVM_ALL_ON)
    `uvm_field_int(size, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(mask, UVM_ALL_ON)
    `uvm_field_int(read_only, UVM_ALL_ON)
    `uvm_field_int(stall, UVM_ALL_ON)
    `uvm_field_int(cancel, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [`SPM_BITS-1:0] value;
  logic error;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(value, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
