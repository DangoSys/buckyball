class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  bank_signals pins[BANKS], sampled[BANKS];
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(ok, UVM_ALL_ON)
    `uvm_field_int(bank, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [WIDTH-1:0] data;
  logic ok;
  int bank;
  bit write;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
