`include "rvv_config.svh"
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
  logic [31:0] done_pc, done_instruction, done_cause;
  logic [63:0] done_tval;
  logic [63:0] done_cycles;
  logic [ 4:0] done_fflags;
  logic
      read_valid[`RVV_MEMORY_PORTS],
      read_ready[`RVV_MEMORY_PORTS],
      read_response_valid[`RVV_MEMORY_PORTS],
      read_response_ready[`RVV_MEMORY_PORTS];
  logic [2:0]
      read_bank[`RVV_MEMORY_PORTS],
      write_bank[`RVV_MEMORY_PORTS],
      read_group[`RVV_MEMORY_PORTS],
      write_group[`RVV_MEMORY_PORTS];
  logic [15:0] read_row[`RVV_MEMORY_PORTS], write_row[`RVV_MEMORY_PORTS];
  logic [127:0] read_data[`RVV_MEMORY_PORTS], write_data[`RVV_MEMORY_PORTS];
  logic
      write_valid[`RVV_MEMORY_PORTS],
      write_ready[`RVV_MEMORY_PORTS],
      write_response_valid[`RVV_MEMORY_PORTS],
      write_response_ready[`RVV_MEMORY_PORTS],
      write_ok[`RVV_MEMORY_PORTS];
  logic [15:0] write_mask[`RVV_MEMORY_PORTS];
  for (genvar p = 0; p < `RVV_MEMORY_PORTS; p++) begin : bank_protocol
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
