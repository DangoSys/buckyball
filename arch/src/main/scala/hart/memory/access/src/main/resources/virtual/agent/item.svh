class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  logic [`VA_INPUT_BITS-1:0] inputs;
  logic [`VA_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(tag, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(line, UVM_ALL_ON)
    `uvm_field_int(misaligned, UVM_ALL_ON)
    `uvm_field_int(page_fault, UVM_ALL_ON)
    `uvm_field_int(access_fault, UVM_ALL_ON)
  `uvm_object_utils_end
  int tag;
  bit [63:0] data;
  bit [`VA_LINE_BITS-1:0] line;
  bit misaligned, page_fault, access_fault;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
