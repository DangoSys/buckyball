class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`BR_INPUT_BITS-1:0] inputs;
  logic [`BR_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(writing, UVM_ALL_ON)
    `uvm_field_int(port, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`BR_DATA_BITS-1:0] data;
  bit writing;
  int port;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
