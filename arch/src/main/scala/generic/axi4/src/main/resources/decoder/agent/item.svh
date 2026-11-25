class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  port_signals ports[3];
  bit reset;
  function new(string name = "cycle");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(value, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [3*$bits(port_signals)-1:0] value;
  function new(string name = "result");
    super.new(name);
  endfunction
endclass
