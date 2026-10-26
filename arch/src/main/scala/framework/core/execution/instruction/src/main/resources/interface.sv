interface port_if (
    input logic clock,
    input logic reset
);
  logic sample;
  logic [31:0] bits;
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
endinterface
