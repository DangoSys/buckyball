class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`TSS_INPUT_BITS-1:0] inputs;
  logic [`TSS_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(value, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
    `uvm_field_int(port, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`TSS_DATA_BITS-1:0] value;
  bit error;
  int port;
  int message_kind;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
