class response extends uvm_sequence_item;
  bit [127:0] data;
  bit [7:0] tag;
  bit error;
  bit write;
  int channel;
  `uvm_object_utils_begin(response)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(tag, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
    `uvm_field_int(write, UVM_ALL_ON)
    `uvm_field_int(channel, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response");
    super.new(name);
  endfunction
endclass
