class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`TM_INPUT_BITS-1:0] inputs;
  logic [`TM_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(value, UVM_ALL_ON)
  `uvm_object_utils_end
  bit [511:0] value;
  function new(string name = "result");
    super.new(name);
  endfunction
endclass
