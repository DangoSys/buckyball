interface rvv_if (
    input logic clock
);
  logic reset;
  logic command_valid, command_ready;
  logic [6:0] command_funct7;
  logic [63:0] command_rs1, command_rs2;
  logic [3:0] command_rob;
  logic image_valid, image_ready;
  logic [31:0] image_data;
  logic busy, done_valid, done_ready, done_fault, done_vxsat;
  logic [ 3:0] done_rob;
  logic [15:0] done_write_bank;
  logic [31:0] done_pc, done_instruction, done_cause, done_tval;
  logic [63:0] done_cycles;
  logic [ 4:0] done_fflags;
  logic read_valid[4], read_ready[4], read_response_valid[4], read_response_ready[4];
  logic [2:0] read_bank[4], write_bank[4];
  logic [7:0] read_row[4], write_row[4];
  logic [127:0] read_data[4], write_data[4];
  logic
      write_valid[4], write_ready[4], write_response_valid[4], write_response_ready[4], write_ok[4];
  logic [15:0] write_mask[4];
  for (genvar p = 0; p < 4; p++) begin : bank_protocol
    assert property(@(posedge clock) disable iff(reset)
      read_valid[p] && !read_ready[p] |=> read_valid[p] && $stable(
        {read_bank[p], read_row[p]}
    ))
    else $fatal(1, "RVV bank read changed under backpressure on port %0d", p);
    assert property(@(posedge clock) disable iff(reset)
      write_valid[p] && !write_ready[p] |=> write_valid[p] && $stable(
        {write_bank[p], write_row[p], write_data[p], write_mask[p]}
    ))
    else $fatal(1, "RVV bank write changed under backpressure on port %0d", p);
  end
endinterface
