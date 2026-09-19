module tb_top;
  import uvm_pkg::*;
  import gemmini_pkg::*;

  bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) intf ();

  GemminiBall dut (
      .clock(intf.clock),
      .reset(intf.reset),
      .io_channelReady(1'b1),
      .io_cmdReq_ready(intf.cmd_req_ready),
      .io_cmdReq_valid(intf.cmd_req_valid),
      .io_cmdReq_bits_cmd_bid(intf.cmd_req_bits_cmd_bid),
      .io_cmdReq_bits_cmd_funct7(intf.cmd_req_bits_cmd_funct7),
      .io_cmdReq_bits_cmd_iter(intf.cmd_req_bits_cmd_iter),
      .io_cmdReq_bits_cmd_op1_en(intf.cmd_req_bits_cmd_op1_en),
      .io_cmdReq_bits_cmd_op2_en(intf.cmd_req_bits_cmd_op2_en),
      .io_cmdReq_bits_cmd_wr_spad_en(intf.cmd_req_bits_cmd_wr_spad_en),
      .io_cmdReq_bits_cmd_op1_from_spad(intf.cmd_req_bits_cmd_op1_from_spad),
      .io_cmdReq_bits_cmd_op2_from_spad(intf.cmd_req_bits_cmd_op2_from_spad),
      .io_cmdReq_bits_cmd_special(intf.cmd_req_bits_cmd_special),
      .io_cmdReq_bits_cmd_op1_bank(intf.cmd_req_bits_cmd_op1_bank),
      .io_cmdReq_bits_cmd_op2_bank(intf.cmd_req_bits_cmd_op2_bank),
      .io_cmdReq_bits_cmd_wr_bank(intf.cmd_req_bits_cmd_wr_bank),
      .io_cmdReq_bits_cmd_op1_col(intf.cmd_req_bits_cmd_op1_col),
      .io_cmdReq_bits_cmd_op2_col(intf.cmd_req_bits_cmd_op2_col),
      .io_cmdReq_bits_cmd_wr_col(intf.cmd_req_bits_cmd_wr_col),
      .io_cmdReq_bits_cmd_meta_bank(intf.cmd_req_bits_cmd_meta_bank),
      .io_cmdReq_bits_cmd_rs1(intf.cmd_req_bits_cmd_rs1),
      .io_cmdReq_bits_cmd_rs2(intf.cmd_req_bits_cmd_rs2),
      .io_cmdReq_bits_rob_id(intf.cmd_req_bits_rob_id),
      .io_cmdReq_bits_is_sub(intf.cmd_req_bits_is_sub),
      .io_cmdReq_bits_sub_rob_id(intf.cmd_req_bits_sub_rob_id),
      .io_cmdResp_ready(intf.cmd_resp_ready),
      .io_cmdResp_valid(intf.cmd_resp_valid),
      .io_cmdResp_bits_rob_id(intf.cmd_resp_bits_rob_id),
      .io_cmdResp_bits_is_sub(intf.cmd_resp_bits_is_sub),
      .io_cmdResp_bits_sub_rob_id(intf.cmd_resp_bits_sub_rob_id),
      .io_bankRead_0_io_req_ready(intf.bank_read_req_ready[0]),
      .io_bankRead_0_io_req_valid(intf.bank_read_req_valid[0]),
      .io_bankRead_0_io_resp_ready(intf.bank_read_resp_ready[0]),
      .io_bankRead_0_io_resp_valid(intf.bank_read_resp_valid[0]),
      .io_bankRead_0_io_resp_bits_data(intf.bank_read_resp_data[0]),
      .io_bankRead_1_io_req_ready(intf.bank_read_req_ready[1]),
      .io_bankRead_1_io_req_valid(intf.bank_read_req_valid[1]),
      .io_bankRead_1_io_resp_ready(intf.bank_read_resp_ready[1]),
      .io_bankRead_1_io_resp_valid(intf.bank_read_resp_valid[1]),
      .io_bankRead_1_io_resp_bits_data(intf.bank_read_resp_data[1]),
      .io_bankWrite_0_io_req_ready(intf.bank_write_req_ready[0]),
      .io_bankWrite_0_io_req_valid(intf.bank_write_req_valid[0]),
      .io_bankWrite_0_io_resp_ready(intf.bank_write_resp_ready[0]),
      .io_bankWrite_0_io_resp_valid(intf.bank_write_resp_valid[0]),
      .io_bankWrite_0_io_resp_bits_ok(intf.bank_write_resp_ok[0]),
      .io_bankWrite_1_io_req_ready(intf.bank_write_req_ready[1]),
      .io_bankWrite_1_io_req_valid(intf.bank_write_req_valid[1]),
      .io_bankWrite_1_io_resp_ready(intf.bank_write_resp_ready[1]),
      .io_bankWrite_1_io_resp_valid(intf.bank_write_resp_valid[1]),
      .io_bankWrite_1_io_resp_bits_ok(intf.bank_write_resp_ok[1]),
      .io_bankWrite_2_io_req_ready(intf.bank_write_req_ready[2]),
      .io_bankWrite_2_io_req_valid(intf.bank_write_req_valid[2]),
      .io_bankWrite_2_io_resp_ready(intf.bank_write_resp_ready[2]),
      .io_bankWrite_2_io_resp_valid(intf.bank_write_resp_valid[2]),
      .io_bankWrite_2_io_resp_bits_ok(intf.bank_write_resp_ok[2]),
      .io_bankWrite_3_io_req_ready(intf.bank_write_req_ready[3]),
      .io_bankWrite_3_io_req_valid(intf.bank_write_req_valid[3]),
      .io_bankWrite_3_io_resp_ready(intf.bank_write_resp_ready[3]),
      .io_bankWrite_3_io_resp_valid(intf.bank_write_resp_valid[3]),
      .io_bankWrite_3_io_resp_bits_ok(intf.bank_write_resp_ok[3]),
      .io_subRobReq_ready(intf.sub_rob_req_ready)
  );

  initial begin
    intf.clock = 1'b0;
    forever #5 intf.clock = ~intf.clock;
  end

  initial begin
    uvm_config_db#(virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW))::set(null, "*", "vif", intf);
    run_test("gemmini_ball_test");
  end
endmodule
