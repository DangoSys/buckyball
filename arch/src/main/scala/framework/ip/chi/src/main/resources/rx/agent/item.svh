class cycle_item extends uvm_sequence_item;
  `uvm_object_utils(cycle_item)
  link_signals pins, sampled;
  bit reset;
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass
class flit_item extends uvm_sequence_item;
  `uvm_object_utils_begin(flit_item)
    `uvm_field_int(data, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [WIDTH-1:0] data;
  function new(string name = "flit_item");
    super.new(name);
  endfunction
endclass
