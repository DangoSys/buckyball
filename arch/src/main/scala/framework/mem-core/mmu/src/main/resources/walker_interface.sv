interface walker_control_if (
    input logic clock
);
  logic reset;
  logic [3:0] mode;
  logic [43:0] root_ppn;
endinterface
