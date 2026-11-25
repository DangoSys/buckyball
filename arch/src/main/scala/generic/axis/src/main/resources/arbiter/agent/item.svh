class cycle_item extends uvm_sequence_item;
  bit reset;
  logic [1:0] valid;
  logic [`AXIS_DATA_BITS-1:0] data[2];
  logic [`AXIS_DATA_BITS/8-1:0] keep[2];
  logic [1:0] last;
  logic ready;
  logic [1:0] accepted;
  logic output_valid;
  logic output_last;
  `uvm_object_utils_begin(cycle_item)
    `uvm_field_int(reset, UVM_ALL_ON)
    `uvm_field_int(valid, UVM_ALL_ON)
    `uvm_field_sarray_int(data, UVM_ALL_ON)
    `uvm_field_sarray_int(keep, UVM_ALL_ON)
    `uvm_field_int(last, UVM_ALL_ON)
    `uvm_field_int(ready, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "cycle_item");
    super.new(name);
  endfunction
endclass

class result_item extends uvm_sequence_item;
  logic valid;
  logic [1:0] ready;
  logic [`AXIS_DATA_BITS-1:0] data;
  logic [`AXIS_DATA_BITS/8-1:0] keep;
  logic last;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(valid, UVM_ALL_ON)
    `uvm_field_int(ready, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(keep, UVM_ALL_ON)
    `uvm_field_int(last, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
