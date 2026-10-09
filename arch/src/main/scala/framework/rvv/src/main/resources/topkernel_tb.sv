module topkernel_tb;
  import uvm_pkg::*;
  import rvv_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  rvv_if vif (clock);
  logic terminal_valid, terminal_ready, image_load_active;
  logic [3:0] terminal_error;
  logic [63:0] terminal_address, image_load_address;
  logic [31:0] image_words_remaining;
  always_ff @(posedge clock) begin
    if (vif.reset) begin
      terminal_valid <= 0;
      terminal_error <= 0;
      terminal_address <= 0;
      image_load_active <= 0;
      image_words_remaining <= 0;
      image_load_address <= 0;
    end else begin
      if (vif.command_valid && vif.command_ready && vif.command_funct7 == 44) begin
        image_load_active <= vif.command_rs1[19:0] == 0;
        image_words_remaining <= vif.command_rs1[63:30] >> 2;
        image_load_address <= vif.command_rs2;
      end
      if (image_load_active && vif.image_valid && vif.image_ready) begin
        image_words_remaining <= image_words_remaining - 1;
        if (image_words_remaining == 1) begin
          terminal_valid   <= 1;
          terminal_error   <= 0;
          terminal_address <= 0;
        end
      end
      if (image_load_active && vif.done_valid && vif.done_fault && !terminal_valid) begin
        terminal_valid   <= 1;
        terminal_error   <= 10;
        terminal_address <= image_load_address;
      end
      if (terminal_valid && terminal_ready) begin
        terminal_valid <= 0;
        image_load_active <= 0;
      end
    end
  end
  wire ball_request;
  always @(posedge clock)
    if (!vif.reset && ball_request)
      $fatal(1, "TopKernel fixture has no Ball command target");
  KernelEngine dut (
      .clock(clock),
      .reset(vif.reset),
      .io_channelReady(1'b1),
      .io_cmdReq_valid(vif.command_valid),
      .io_cmdReq_ready(vif.command_ready),
      .io_cmdReq_bits_cmd_funct7(vif.command_funct7),
      .io_cmdReq_bits_cmd_rs1(vif.command_rs1),
      .io_cmdReq_bits_cmd_rs2(vif.command_rs2),
      .io_cmdReq_bits_rob_id(vif.command_rob),
      .io_cmdReq_bits_read_groups(3'd1),
      .io_cmdReq_bits_write_groups(3'd5),
      .io_cmdResp_valid(vif.done_valid),
      .io_cmdResp_ready(vif.done_ready),
      .io_cmdResp_bits_rob_id(vif.done_rob),
      .io_cmdResp_bits_write_bank(vif.done_write_bank),
      .io_status_running(vif.busy),
      .io_image_valid(vif.image_valid),
      .io_image_ready(vif.image_ready),
      .io_image_bits(vif.image_data),
      .io_imageTerminal_valid(terminal_valid),
      .io_imageTerminal_ready(terminal_ready),
      .io_imageTerminal_bits_error(terminal_error),
      .io_imageTerminal_bits_address(terminal_address),
      .io_ballRequest_ready(1'b0),
      .io_ballRequest_valid(ball_request),
      .io_ballResponse_valid(1'b0),
      .io_ballResponse_bits(64'b0),
      .io_result_fault(vif.done_fault),
      .io_result_pc(vif.done_pc),
      .io_result_instruction(vif.done_instruction),
      .io_result_cycles(vif.done_cycles),
      .io_result_cause(vif.done_cause),
      .io_result_tval(vif.done_tval),
      .io_result_fflags(vif.done_fflags),
      .io_result_vxsat(vif.done_vxsat),
      `include "rvv_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual rvv_if)::set(null, "uvm_test_top*", "vif", vif);
    run_test("protocol_test");
  end
endmodule
