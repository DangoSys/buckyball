class request_item extends uvm_sequence_item;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [4:0] fn;
  logic dw;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(a, UVM_ALL_ON)
    `uvm_field_int(rhs, UVM_ALL_ON)
    `uvm_field_int(fn, UVM_ALL_ON)
    `uvm_field_int(dw, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [63:0] value;
  logic [63:0] adder;
  logic comparison;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(value, UVM_ALL_ON)
    `uvm_field_int(adder, UVM_ALL_ON)
    `uvm_field_int(comparison, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
