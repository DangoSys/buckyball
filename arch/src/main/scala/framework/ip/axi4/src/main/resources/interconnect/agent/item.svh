class batch extends uvm_sequence_item;
  `uvm_object_utils(batch)
  int unsigned serial;
  int unsigned beats;
  bit stalls;
  function new(string name = "batch");
    super.new(name);
  endfunction
endclass

class snapshot extends uvm_sequence_item;
  `uvm_object_utils(snapshot)
  port_signals ports[M+1];
  int unsigned outstanding;
  bit reset;
  function new(string name = "sample");
    super.new(name);
  endfunction
endclass

class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(payload, UVM_ALL_ON)
    `uvm_field_int(channel, UVM_ALL_ON)
    `uvm_field_int(owner, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [D+128:0] payload;
  int channel;
  int owner;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
