class request_item extends uvm_sequence_item;
  logic [3:0] addr;
  logic write;
  logic [31:0] data;
  logic [3:0] mask;
  logic [7:0] tag;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(addr, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(mask, UVM_ALL_ON)
    `uvm_field_int(tag, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [31:0] data;
  logic [7:0] tag;
  logic error;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(tag, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
