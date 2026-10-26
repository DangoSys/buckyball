class request_item extends uvm_sequence_item;
  logic [31:0] bits;
  `uvm_object_utils_begin(request_item)
    `uvm_field_int(bits, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [31:0] expanded;
  logic compressed;
  logic illegal;
  logic [6:0] opcode;
  logic [2:0] funct3;
  logic [6:0] funct7;
  logic [4:0] rs1;
  logic [4:0] rs2;
  logic [4:0] rd;
  logic [11:0] csr_address;
  logic [63:0] immediate;
  logic [3:0] atomic;
  logic [4:0] function_code;
  logic word;
  logic immediate_mode;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(expanded, UVM_ALL_ON)
    `uvm_field_int(compressed, UVM_ALL_ON)
    `uvm_field_int(illegal, UVM_ALL_ON)
    `uvm_field_int(opcode, UVM_ALL_ON)
    `uvm_field_int(funct3, UVM_ALL_ON)
    `uvm_field_int(funct7, UVM_ALL_ON)
    `uvm_field_int(rs1, UVM_ALL_ON)
    `uvm_field_int(rs2, UVM_ALL_ON)
    `uvm_field_int(rd, UVM_ALL_ON)
    `uvm_field_int(csr_address, UVM_ALL_ON)
    `uvm_field_int(immediate, UVM_ALL_ON)
    `uvm_field_int(atomic, UVM_ALL_ON)
    `uvm_field_int(function_code, UVM_ALL_ON)
    `uvm_field_int(word, UVM_ALL_ON)
    `uvm_field_int(immediate_mode, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
