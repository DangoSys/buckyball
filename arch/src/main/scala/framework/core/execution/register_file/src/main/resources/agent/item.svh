class request_item extends uvm_sequence_item;
  logic [4:0] read0;
  logic [4:0] read1;
  logic write;
  logic [4:0] address;
  logic [63:0] data;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(read0, UVM_ALL_ON)
    `uvm_field_int(read1, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(address, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [63:0] data0;
  logic [63:0] data1;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(data0, UVM_ALL_ON)
    `uvm_field_int(data1, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
