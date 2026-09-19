class item extends uvm_sequence_item;
  rand bit [31:0] data;
  rand bit [ 3:0] keep;
  rand bit        last;

  `uvm_object_utils_begin(item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(keep, UVM_ALL_ON)
    `uvm_field_int(last, UVM_ALL_ON)
  `uvm_object_utils_end

  function new(string name = "item");
    super.new(name);
  endfunction
endclass
