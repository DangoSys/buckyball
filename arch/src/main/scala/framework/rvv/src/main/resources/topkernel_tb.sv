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
      if (vif.command_valid && vif.command_ready && vif.command_funct7 == 12) begin
        image_load_active <= vif.command_rs1[63:33] == 0;
        image_words_remaining <= vif.command_rs1[31:0] >> 2;
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
      .io_cmdReq_bits_cmd_bid('0),
      .io_result_fault(vif.done_fault),
      .io_result_pc(vif.done_pc),
      .io_result_instruction(vif.done_instruction),
      .io_result_cycles(vif.done_cycles),
      .io_result_cause(vif.done_cause),
      .io_result_tval(vif.done_tval),
      .io_result_fflags(vif.done_fflags),
      .io_result_vxsat(vif.done_vxsat),
      .io_bankRead_0_bank_id(vif.read_bank[0]),
      .io_bankRead_0_io_req_valid(vif.read_valid[0]),
      .io_bankRead_0_io_req_ready(vif.read_ready[0]),
      .io_bankRead_0_io_req_bits_addr(vif.read_row[0]),
      .io_bankRead_0_io_resp_valid(vif.read_response_valid[0]),
      .io_bankRead_0_io_resp_ready(vif.read_response_ready[0]),
      .io_bankRead_0_io_resp_bits_data(vif.read_data[0]),
      .io_bankWrite_0_bank_id(vif.write_bank[0]),
      .io_bankWrite_0_io_req_valid(vif.write_valid[0]),
      .io_bankWrite_0_io_req_ready(vif.write_ready[0]),
      .io_bankWrite_0_io_req_bits_addr(vif.write_row[0]),
      .io_bankWrite_0_io_resp_valid(vif.write_response_valid[0]),
      .io_bankWrite_0_io_resp_ready(vif.write_response_ready[0]),
      .io_bankWrite_0_io_req_bits_data(vif.write_data[0]),
      .io_bankWrite_0_io_resp_bits_ok(vif.write_ok[0]),
      .io_bankWrite_0_io_req_bits_mask_0(vif.write_mask[0][0]),
      .io_bankWrite_0_io_req_bits_mask_1(vif.write_mask[0][1]),
      .io_bankWrite_0_io_req_bits_mask_2(vif.write_mask[0][2]),
      .io_bankWrite_0_io_req_bits_mask_3(vif.write_mask[0][3]),
      .io_bankWrite_0_io_req_bits_mask_4(vif.write_mask[0][4]),
      .io_bankWrite_0_io_req_bits_mask_5(vif.write_mask[0][5]),
      .io_bankWrite_0_io_req_bits_mask_6(vif.write_mask[0][6]),
      .io_bankWrite_0_io_req_bits_mask_7(vif.write_mask[0][7]),
      .io_bankWrite_0_io_req_bits_mask_8(vif.write_mask[0][8]),
      .io_bankWrite_0_io_req_bits_mask_9(vif.write_mask[0][9]),
      .io_bankWrite_0_io_req_bits_mask_10(vif.write_mask[0][10]),
      .io_bankWrite_0_io_req_bits_mask_11(vif.write_mask[0][11]),
      .io_bankWrite_0_io_req_bits_mask_12(vif.write_mask[0][12]),
      .io_bankWrite_0_io_req_bits_mask_13(vif.write_mask[0][13]),
      .io_bankWrite_0_io_req_bits_mask_14(vif.write_mask[0][14]),
      .io_bankWrite_0_io_req_bits_mask_15(vif.write_mask[0][15]),
      .io_bankRead_1_bank_id(vif.read_bank[1]),
      .io_bankRead_1_io_req_valid(vif.read_valid[1]),
      .io_bankRead_1_io_req_ready(vif.read_ready[1]),
      .io_bankRead_1_io_req_bits_addr(vif.read_row[1]),
      .io_bankRead_1_io_resp_valid(vif.read_response_valid[1]),
      .io_bankRead_1_io_resp_ready(vif.read_response_ready[1]),
      .io_bankRead_1_io_resp_bits_data(vif.read_data[1]),
      .io_bankWrite_1_bank_id(vif.write_bank[1]),
      .io_bankWrite_1_io_req_valid(vif.write_valid[1]),
      .io_bankWrite_1_io_req_ready(vif.write_ready[1]),
      .io_bankWrite_1_io_req_bits_addr(vif.write_row[1]),
      .io_bankWrite_1_io_resp_valid(vif.write_response_valid[1]),
      .io_bankWrite_1_io_resp_ready(vif.write_response_ready[1]),
      .io_bankWrite_1_io_req_bits_data(vif.write_data[1]),
      .io_bankWrite_1_io_resp_bits_ok(vif.write_ok[1]),
      .io_bankWrite_1_io_req_bits_mask_0(vif.write_mask[1][0]),
      .io_bankWrite_1_io_req_bits_mask_1(vif.write_mask[1][1]),
      .io_bankWrite_1_io_req_bits_mask_2(vif.write_mask[1][2]),
      .io_bankWrite_1_io_req_bits_mask_3(vif.write_mask[1][3]),
      .io_bankWrite_1_io_req_bits_mask_4(vif.write_mask[1][4]),
      .io_bankWrite_1_io_req_bits_mask_5(vif.write_mask[1][5]),
      .io_bankWrite_1_io_req_bits_mask_6(vif.write_mask[1][6]),
      .io_bankWrite_1_io_req_bits_mask_7(vif.write_mask[1][7]),
      .io_bankWrite_1_io_req_bits_mask_8(vif.write_mask[1][8]),
      .io_bankWrite_1_io_req_bits_mask_9(vif.write_mask[1][9]),
      .io_bankWrite_1_io_req_bits_mask_10(vif.write_mask[1][10]),
      .io_bankWrite_1_io_req_bits_mask_11(vif.write_mask[1][11]),
      .io_bankWrite_1_io_req_bits_mask_12(vif.write_mask[1][12]),
      .io_bankWrite_1_io_req_bits_mask_13(vif.write_mask[1][13]),
      .io_bankWrite_1_io_req_bits_mask_14(vif.write_mask[1][14]),
      .io_bankWrite_1_io_req_bits_mask_15(vif.write_mask[1][15]),
      .io_bankRead_2_bank_id(vif.read_bank[2]),
      .io_bankRead_2_io_req_valid(vif.read_valid[2]),
      .io_bankRead_2_io_req_ready(vif.read_ready[2]),
      .io_bankRead_2_io_req_bits_addr(vif.read_row[2]),
      .io_bankRead_2_io_resp_valid(vif.read_response_valid[2]),
      .io_bankRead_2_io_resp_ready(vif.read_response_ready[2]),
      .io_bankRead_2_io_resp_bits_data(vif.read_data[2]),
      .io_bankWrite_2_bank_id(vif.write_bank[2]),
      .io_bankWrite_2_io_req_valid(vif.write_valid[2]),
      .io_bankWrite_2_io_req_ready(vif.write_ready[2]),
      .io_bankWrite_2_io_req_bits_addr(vif.write_row[2]),
      .io_bankWrite_2_io_resp_valid(vif.write_response_valid[2]),
      .io_bankWrite_2_io_resp_ready(vif.write_response_ready[2]),
      .io_bankWrite_2_io_req_bits_data(vif.write_data[2]),
      .io_bankWrite_2_io_resp_bits_ok(vif.write_ok[2]),
      .io_bankWrite_2_io_req_bits_mask_0(vif.write_mask[2][0]),
      .io_bankWrite_2_io_req_bits_mask_1(vif.write_mask[2][1]),
      .io_bankWrite_2_io_req_bits_mask_2(vif.write_mask[2][2]),
      .io_bankWrite_2_io_req_bits_mask_3(vif.write_mask[2][3]),
      .io_bankWrite_2_io_req_bits_mask_4(vif.write_mask[2][4]),
      .io_bankWrite_2_io_req_bits_mask_5(vif.write_mask[2][5]),
      .io_bankWrite_2_io_req_bits_mask_6(vif.write_mask[2][6]),
      .io_bankWrite_2_io_req_bits_mask_7(vif.write_mask[2][7]),
      .io_bankWrite_2_io_req_bits_mask_8(vif.write_mask[2][8]),
      .io_bankWrite_2_io_req_bits_mask_9(vif.write_mask[2][9]),
      .io_bankWrite_2_io_req_bits_mask_10(vif.write_mask[2][10]),
      .io_bankWrite_2_io_req_bits_mask_11(vif.write_mask[2][11]),
      .io_bankWrite_2_io_req_bits_mask_12(vif.write_mask[2][12]),
      .io_bankWrite_2_io_req_bits_mask_13(vif.write_mask[2][13]),
      .io_bankWrite_2_io_req_bits_mask_14(vif.write_mask[2][14]),
      .io_bankWrite_2_io_req_bits_mask_15(vif.write_mask[2][15]),
      .io_bankRead_3_bank_id(vif.read_bank[3]),
      .io_bankRead_3_io_req_valid(vif.read_valid[3]),
      .io_bankRead_3_io_req_ready(vif.read_ready[3]),
      .io_bankRead_3_io_req_bits_addr(vif.read_row[3]),
      .io_bankRead_3_io_resp_valid(vif.read_response_valid[3]),
      .io_bankRead_3_io_resp_ready(vif.read_response_ready[3]),
      .io_bankRead_3_io_resp_bits_data(vif.read_data[3]),
      .io_bankWrite_3_bank_id(vif.write_bank[3]),
      .io_bankWrite_3_io_req_valid(vif.write_valid[3]),
      .io_bankWrite_3_io_req_ready(vif.write_ready[3]),
      .io_bankWrite_3_io_req_bits_addr(vif.write_row[3]),
      .io_bankWrite_3_io_resp_valid(vif.write_response_valid[3]),
      .io_bankWrite_3_io_resp_ready(vif.write_response_ready[3]),
      .io_bankWrite_3_io_req_bits_data(vif.write_data[3]),
      .io_bankWrite_3_io_resp_bits_ok(vif.write_ok[3]),
      .io_bankWrite_3_io_req_bits_mask_0(vif.write_mask[3][0]),
      .io_bankWrite_3_io_req_bits_mask_1(vif.write_mask[3][1]),
      .io_bankWrite_3_io_req_bits_mask_2(vif.write_mask[3][2]),
      .io_bankWrite_3_io_req_bits_mask_3(vif.write_mask[3][3]),
      .io_bankWrite_3_io_req_bits_mask_4(vif.write_mask[3][4]),
      .io_bankWrite_3_io_req_bits_mask_5(vif.write_mask[3][5]),
      .io_bankWrite_3_io_req_bits_mask_6(vif.write_mask[3][6]),
      .io_bankWrite_3_io_req_bits_mask_7(vif.write_mask[3][7]),
      .io_bankWrite_3_io_req_bits_mask_8(vif.write_mask[3][8]),
      .io_bankWrite_3_io_req_bits_mask_9(vif.write_mask[3][9]),
      .io_bankWrite_3_io_req_bits_mask_10(vif.write_mask[3][10]),
      .io_bankWrite_3_io_req_bits_mask_11(vif.write_mask[3][11]),
      .io_bankWrite_3_io_req_bits_mask_12(vif.write_mask[3][12]),
      .io_bankWrite_3_io_req_bits_mask_13(vif.write_mask[3][13]),
      .io_bankWrite_3_io_req_bits_mask_14(vif.write_mask[3][14]),
      .io_bankWrite_3_io_req_bits_mask_15(vif.write_mask[3][15])
  );
  initial begin
    uvm_config_db#(virtual rvv_if)::set(null, "uvm_test_top*", "vif", vif);
    run_test("protocol_test");
  end
endmodule
