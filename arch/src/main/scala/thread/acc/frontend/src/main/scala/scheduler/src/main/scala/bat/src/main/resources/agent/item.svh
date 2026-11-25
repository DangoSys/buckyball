class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`BAT_INPUT_BITS-1:0] inputs;
  logic [`BAT_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(payload, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`BAT_OUTPUT_BITS-1:0] payload;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
