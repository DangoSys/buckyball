class packet_item extends uvm_sequence_item;
  `uvm_object_utils(packet_item)
  logic [`CHI_DAT_BITS-1:0] payload;
  bit sc_success, failed;
  function new(string name = "packet_item");
    super.new(name);
  endfunction
endclass
class sample_item extends uvm_sequence_item;
  `uvm_object_utils(sample_item)
  pins_t sampled;
  function new(string name = "sample_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(error, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [63:0] data;
  logic error;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
class packet_sequence extends uvm_sequence #(packet_item);
  `uvm_object_utils(packet_sequence)
  packet_item item;
  function new(string name = "packet_sequence");
    super.new(name);
  endfunction
  task body();
    start_item(item);
    finish_item(item);
  endtask
endclass
