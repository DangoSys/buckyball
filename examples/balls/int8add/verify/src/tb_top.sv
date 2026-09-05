module tb_top;
  import uvm_pkg::*;
  import int8add_pkg::*;

  bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) intf ();

  Int8AddBall dut (
      .clock(intf.clock),
      .reset(intf.reset),
      .io_status_idle(),
      .io_status_running(),
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
      .io_bankRead_0_bank_id(intf.bank_read_bank_id[0]),
      .io_bankRead_0_rob_id(intf.bank_read_rob_id[0]),
      .io_bankRead_0_ball_id(),
      .io_bankRead_0_group_id(intf.bank_read_group_id[0]),
      .io_bankRead_0_io_req_ready(intf.bank_read_req_ready[0]),
      .io_bankRead_0_io_req_valid(intf.bank_read_req_valid[0]),
      .io_bankRead_0_io_req_bits_addr(intf.bank_read_req_addr[0][7:0]),
      .io_bankRead_0_io_resp_ready(intf.bank_read_resp_ready[0]),
      .io_bankRead_0_io_resp_valid(intf.bank_read_resp_valid[0]),
      .io_bankRead_0_io_resp_bits_data(intf.bank_read_resp_data[0]),
      .io_bankRead_1_bank_id(intf.bank_read_bank_id[1]),
      .io_bankRead_1_rob_id(intf.bank_read_rob_id[1]),
      .io_bankRead_1_ball_id(),
      .io_bankRead_1_group_id(intf.bank_read_group_id[1]),
      .io_bankRead_1_io_req_ready(intf.bank_read_req_ready[1]),
      .io_bankRead_1_io_req_valid(intf.bank_read_req_valid[1]),
      .io_bankRead_1_io_req_bits_addr(intf.bank_read_req_addr[1][7:0]),
      .io_bankRead_1_io_resp_ready(intf.bank_read_resp_ready[1]),
      .io_bankRead_1_io_resp_valid(intf.bank_read_resp_valid[1]),
      .io_bankRead_1_io_resp_bits_data(intf.bank_read_resp_data[1]),
      .io_bankWrite_0_bank_id(intf.bank_write_bank_id[0]),
      .io_bankWrite_0_rob_id(intf.bank_write_rob_id[0]),
      .io_bankWrite_0_ball_id(),
      .io_bankWrite_0_group_id(intf.bank_write_group_id[0]),
      .io_bankWrite_0_io_req_ready(intf.bank_write_req_ready[0]),
      .io_bankWrite_0_io_req_valid(intf.bank_write_req_valid[0]),
      .io_bankWrite_0_io_req_bits_addr(intf.bank_write_req_addr[0][7:0]),
      .io_bankWrite_0_io_req_bits_mask_0(intf.bank_write_req_mask[0][0]),
      .io_bankWrite_0_io_req_bits_mask_1(intf.bank_write_req_mask[0][1]),
      .io_bankWrite_0_io_req_bits_mask_2(intf.bank_write_req_mask[0][2]),
      .io_bankWrite_0_io_req_bits_mask_3(intf.bank_write_req_mask[0][3]),
      .io_bankWrite_0_io_req_bits_mask_4(intf.bank_write_req_mask[0][4]),
      .io_bankWrite_0_io_req_bits_mask_5(intf.bank_write_req_mask[0][5]),
      .io_bankWrite_0_io_req_bits_mask_6(intf.bank_write_req_mask[0][6]),
      .io_bankWrite_0_io_req_bits_mask_7(intf.bank_write_req_mask[0][7]),
      .io_bankWrite_0_io_req_bits_mask_8(intf.bank_write_req_mask[0][8]),
      .io_bankWrite_0_io_req_bits_mask_9(intf.bank_write_req_mask[0][9]),
      .io_bankWrite_0_io_req_bits_mask_10(intf.bank_write_req_mask[0][10]),
      .io_bankWrite_0_io_req_bits_mask_11(intf.bank_write_req_mask[0][11]),
      .io_bankWrite_0_io_req_bits_mask_12(intf.bank_write_req_mask[0][12]),
      .io_bankWrite_0_io_req_bits_mask_13(intf.bank_write_req_mask[0][13]),
      .io_bankWrite_0_io_req_bits_mask_14(intf.bank_write_req_mask[0][14]),
      .io_bankWrite_0_io_req_bits_mask_15(intf.bank_write_req_mask[0][15]),
      .io_bankWrite_0_io_req_bits_data(intf.bank_write_req_data[0]),
      .io_bankWrite_0_io_resp_ready(intf.bank_write_resp_ready[0]),
      .io_bankWrite_0_io_resp_valid(intf.bank_write_resp_valid[0]),
      .io_bankWrite_0_io_resp_bits_ok(intf.bank_write_resp_ok[0]),
      .io_subRobReq_ready(intf.sub_rob_req_ready)
  );

  initial begin
    intf.clock = 1'b0;
    intf.reset = 1'b1;
    forever #5 intf.clock = ~intf.clock;
  end

  initial begin
    uvm_config_db#(virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW))::set(null, "*", "vif", intf);
    run_test("int8add_ball_test");
  end
endmodule
