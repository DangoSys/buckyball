typedef logic [`MMU_REQ_BITS-1:0] request_t;
class request_item extends uvm_sequence_item;
  `uvm_object_utils(request_item)
  request_t payload;
  int kind = 0;
  int mode;
  longint unsigned root;
  bit change_inputs = 0;
  int next_mode;
  longint unsigned next_root;
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass
class request_sequence extends uvm_sequence #(request_item);
  `uvm_object_utils(request_sequence)
  request_item item;
  function new(string name = "request_sequence");
    super.new(name);
  endfunction
  task body();
    start_item(item);
    finish_item(item);
  endtask
endclass
class sample_item extends uvm_sequence_item;
  `uvm_object_utils(sample_item)
  logic [`MMU_INPUT_BITS-1:0] inputs;
  logic [`MMU_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "sample_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(payload, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`MMU_RESP_BITS-1:0] payload;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
