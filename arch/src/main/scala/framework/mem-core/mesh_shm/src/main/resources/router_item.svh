class packet extends uvm_sequence_item;
  bit [127:0] data;
  bit [15:0] keep;
  bit last;
  bit [7:0] id;
  bit [3:0] destination;
  bit [39:0] events;
  int port;
  `uvm_object_utils_begin(packet)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(keep, UVM_ALL_ON)
    `uvm_field_int(last, UVM_ALL_ON)
    `uvm_field_int(id, UVM_ALL_ON)
    `uvm_field_int(destination, UVM_ALL_ON)
    `uvm_field_int(events, UVM_ALL_ON)
    `uvm_field_int(port, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "packet");
    super.new(name);
  endfunction
endclass
