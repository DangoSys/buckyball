class request_item extends uvm_sequence_item;
  logic [31:0] instruction;
  logic [63:0] integer_value;
  logic [63:0] a;
  logic [63:0] rhs;
  logic [63:0] c;
  logic [2:0] rounding;
  int unsigned stall;
  logic cancel;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(instruction, UVM_ALL_ON)
    `uvm_field_int(integer_value, UVM_ALL_ON)
    `uvm_field_int(a, UVM_ALL_ON)
    `uvm_field_int(rhs, UVM_ALL_ON)
    `uvm_field_int(c, UVM_ALL_ON)
    `uvm_field_int(rounding, UVM_ALL_ON)
    `uvm_field_int(stall, UVM_ALL_ON)
    `uvm_field_int(cancel, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [63:0] value;
  logic [4:0] flags;
  logic integer_result;
  logic illegal;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(value, UVM_ALL_ON)
    `uvm_field_int(flags, UVM_ALL_ON)
    `uvm_field_int(integer_result, UVM_ALL_ON)
    `uvm_field_int(illegal, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
