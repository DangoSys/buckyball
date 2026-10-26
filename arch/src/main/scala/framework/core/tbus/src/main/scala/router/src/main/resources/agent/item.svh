class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`TR_INPUT_BITS-1:0] inputs;
  logic [`TR_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
    `uvm_field_int(kind, UVM_ALL_ON)
    `uvm_field_int(port, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [127:0] data;
  bit error;
  int kind, port;
  function new(string name = "result");
    super.new(name);
  endfunction
endclass
