class request_item extends uvm_sequence_item;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [2:0] operation;
  logic word;
  int unsigned stall;
  logic cancel;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(a, UVM_ALL_ON)
    `uvm_field_int(rhs, UVM_ALL_ON)
    `uvm_field_int(operation, UVM_ALL_ON)
    `uvm_field_int(word, UVM_ALL_ON)
    `uvm_field_int(stall, UVM_ALL_ON)
    `uvm_field_int(cancel, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [63:0] value;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(value, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
