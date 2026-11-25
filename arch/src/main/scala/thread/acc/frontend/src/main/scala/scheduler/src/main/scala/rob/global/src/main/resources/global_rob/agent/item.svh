typedef logic [`GROB_IN_ALLOC_BITS-1:0] command_t;
typedef logic [`GROB_IN_OPERATION_BITS-1:0] operation_t;
typedef logic [`GROB_IN_RENAMED_BITS-1:0] bank_access_t;
typedef logic [`GROB_ISSUE_BITS-1:0] issue_t;
class command_item extends uvm_sequence_item;
  `uvm_object_utils(command_item)
  command_t payload;
  function new(string name = "command_item");
    super.new(name);
  endfunction
endclass
class command_sequence extends uvm_sequence #(command_item);
  `uvm_object_utils(command_sequence)
  command_item item;
  function new(string name = "command_sequence");
    super.new(name);
  endfunction
  task body();
    start_item(item);
    finish_item(item);
  endtask
endclass
class sample_item extends uvm_sequence_item;
  `uvm_object_utils(sample_item)
  logic [`GROB_INPUT_BITS-1:0] inputs;
  logic [`GROB_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "sample_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(payload, UVM_ALL_ON)
  `uvm_object_utils_end
  issue_t payload;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
class transaction;
  operation_t operation;
  issue_t expected;
  bank_access_t renamed;
  int slot, serial, raw_write, new_alias, stale_raw, funct;
  longint unsigned pc, inst_id;
  bit [1:0] stale_valid;
  int stale[2];
  bit selected = 0, delivered = 0, completed = 0, allocated = 0, config_op = 0, movement = 0;
  bit [`GROB_VBANKS-1:0] reads, writes;
  int due;
endclass
