typedef enum int {
  REDIRECT,
  FENCE,
  CONTEXT,
  INVALIDATE,
  RESET_VECTOR,
  BAD_FLUSH
} control_kind;
class program_image;
  int version = 0;
endclass
class control_item extends uvm_sequence_item;
  `uvm_object_utils(control_item)
  control_kind kind;
  longint unsigned target, satp;
  int privilege, version;
  bit sum, mxr;
  function new(string name = "control_item");
    super.new(name);
  endfunction
endclass
class control_sequence extends uvm_sequence #(control_item);
  `uvm_object_utils(control_sequence)
  control_item item;
  function new(string name = "control_sequence");
    super.new(name);
  endfunction
  task body();
    start_item(item);
    finish_item(item);
  endtask
endclass
class sample_item extends uvm_sequence_item;
  `uvm_object_utils(sample_item)
  logic [`FETCH_INPUT_BITS-1:0] inputs;
  logic [`FETCH_OUTPUT_BITS-1:0] outputs;
  bit reset;
  function new(string name = "sample_item");
    super.new(name);
  endfunction
endclass
class result_item extends uvm_sequence_item;
  `uvm_object_utils_begin(result_item)
    `uvm_field_int(payload, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`FETCH_PACKET_BITS-1:0] payload;
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
class word_transaction;
  longint unsigned pc, address;
  int epoch, version, due;
  logic [63:0] data;
  logic [511:0] line;
  bit executable, pf, af;
endclass
