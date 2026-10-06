interface mesh_fabric_if (
    input logic clock
);
  logic reset, active;
  logic valid[5][4], ready[5][4], out_valid[5][4], out_ready[5][4];
  logic [1:0] x[5][4], y[5][4], src_x[5][4], src_y[5][4], vc[5][4];
  logic head[5][4], tail[5][4];
  logic [31:0] data[5][4], out_data[5][4];
  logic [1:0] out_src_x[5][4], out_src_y[5][4];
  logic out_head[5][4], out_tail[5][4];
endinterface
