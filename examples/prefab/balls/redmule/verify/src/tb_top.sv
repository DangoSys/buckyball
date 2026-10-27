module tb_top;
  import uvm_pkg::*;
  import redmule_uvm_pkg::*;

  bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) intf ();

  redmule_reference reference_model (
      intf.clock,
      intf.reset
  );
  redmule_reference_if reference_interface (
      intf.clock,
      intf.reset
  );

  RedmuleBall dut (
      .clock(intf.clock),
      .reset(intf.reset),
      .blink_status_running(intf.status_running),
      .blink_cmdReq_ready(intf.cmd_req_ready),
      .blink_cmdReq_valid(intf.cmd_req_valid),
      .blink_cmdReq_bits_rob_id(intf.cmd_req_bits_rob_id),
      .blink_cmdReq_bits_funct(intf.cmd_req_bits_funct),
      .blink_cmdReq_bits_rs2(intf.cmd_req_bits_rs2),
      .blink_cmdReq_bits_iter(intf.cmd_req_bits_iter),
      .blink_cmdReq_bits_op1_bank(intf.cmd_req_bits_op1_bank),
      .blink_cmdReq_bits_op2_bank(intf.cmd_req_bits_op2_bank),
      .blink_cmdReq_bits_wr_bank(intf.cmd_req_bits_wr_bank),
      .blink_cmdReq_bits_op1_group(intf.cmd_req_bits_op1_group),
      .blink_cmdReq_bits_op2_group(intf.cmd_req_bits_op2_group),
      .blink_cmdReq_bits_wr_group(intf.cmd_req_bits_wr_group),
      .blink_cmdReq_bits_op1_col(intf.cmd_req_bits_op1_col),
      .blink_cmdReq_bits_op2_col(intf.cmd_req_bits_op2_col),
      .blink_cmdReq_bits_wr_col(intf.cmd_req_bits_wr_col),
      .blink_cmdReq_bits_wr_read_bank(intf.cmd_req_bits_wr_read_bank),
      .blink_cmdReq_bits_wr_arch_bank(intf.cmd_req_bits_wr_arch_bank),
      .blink_cmdResp_ready(intf.cmd_resp_ready),
      .blink_cmdResp_valid(intf.cmd_resp_valid),
      .blink_cmdResp_bits_rob_id(intf.cmd_resp_bits_rob_id),
      .blink_bankRead_0_bank_id(intf.bank_read_bank_id[0]),
      .blink_bankRead_0_rob_id(intf.bank_read_rob_id[0]),
      .blink_bankRead_0_group_id(intf.bank_read_group_id[0]),
      .blink_bankRead_0_io_req_ready(intf.bank_read_req_ready[0]),
      .blink_bankRead_0_io_req_valid(intf.bank_read_req_valid[0]),
      .blink_bankRead_0_io_req_bits_addr(intf.bank_read_req_addr[0]),
      .blink_bankRead_0_io_resp_ready(intf.bank_read_resp_ready[0]),
      .blink_bankRead_0_io_resp_valid(intf.bank_read_resp_valid[0]),
      .blink_bankRead_0_io_resp_bits_data(intf.bank_read_resp_data[0]),
      .blink_bankRead_1_bank_id(intf.bank_read_bank_id[1]),
      .blink_bankRead_1_rob_id(intf.bank_read_rob_id[1]),
      .blink_bankRead_1_group_id(intf.bank_read_group_id[1]),
      .blink_bankRead_1_io_req_ready(intf.bank_read_req_ready[1]),
      .blink_bankRead_1_io_req_valid(intf.bank_read_req_valid[1]),
      .blink_bankRead_1_io_req_bits_addr(intf.bank_read_req_addr[1]),
      .blink_bankRead_1_io_resp_ready(intf.bank_read_resp_ready[1]),
      .blink_bankRead_1_io_resp_valid(intf.bank_read_resp_valid[1]),
      .blink_bankRead_1_io_resp_bits_data(intf.bank_read_resp_data[1]),
      .blink_bankWrite_0_bank_id(intf.bank_write_bank_id[0]),
      .blink_bankWrite_0_rob_id(intf.bank_write_rob_id[0]),
      .blink_bankWrite_0_group_id(intf.bank_write_group_id[0]),
      .blink_bankWrite_0_io_req_ready(intf.bank_write_req_ready[0]),
      .blink_bankWrite_0_io_req_valid(intf.bank_write_req_valid[0]),
      .blink_bankWrite_0_io_req_bits_addr(intf.bank_write_req_addr[0]),
      .blink_bankWrite_0_io_req_bits_mask_0(intf.bank_write_req_mask[0][0]),
      .blink_bankWrite_0_io_req_bits_mask_1(intf.bank_write_req_mask[0][1]),
      .blink_bankWrite_0_io_req_bits_mask_2(intf.bank_write_req_mask[0][2]),
      .blink_bankWrite_0_io_req_bits_mask_3(intf.bank_write_req_mask[0][3]),
      .blink_bankWrite_0_io_req_bits_mask_4(intf.bank_write_req_mask[0][4]),
      .blink_bankWrite_0_io_req_bits_mask_5(intf.bank_write_req_mask[0][5]),
      .blink_bankWrite_0_io_req_bits_mask_6(intf.bank_write_req_mask[0][6]),
      .blink_bankWrite_0_io_req_bits_mask_7(intf.bank_write_req_mask[0][7]),
      .blink_bankWrite_0_io_req_bits_mask_8(intf.bank_write_req_mask[0][8]),
      .blink_bankWrite_0_io_req_bits_mask_9(intf.bank_write_req_mask[0][9]),
      .blink_bankWrite_0_io_req_bits_mask_10(intf.bank_write_req_mask[0][10]),
      .blink_bankWrite_0_io_req_bits_mask_11(intf.bank_write_req_mask[0][11]),
      .blink_bankWrite_0_io_req_bits_mask_12(intf.bank_write_req_mask[0][12]),
      .blink_bankWrite_0_io_req_bits_mask_13(intf.bank_write_req_mask[0][13]),
      .blink_bankWrite_0_io_req_bits_mask_14(intf.bank_write_req_mask[0][14]),
      .blink_bankWrite_0_io_req_bits_mask_15(intf.bank_write_req_mask[0][15]),
      .blink_bankWrite_0_io_req_bits_data(intf.bank_write_req_data[0]),
      .blink_bankWrite_0_io_resp_ready(intf.bank_write_resp_ready[0]),
      .blink_bankWrite_0_io_resp_valid(intf.bank_write_resp_valid[0]),
      .blink_bankWrite_0_io_resp_bits_ok(intf.bank_write_resp_ok[0])
  );

  initial begin
    intf.clock = 1'b0;
    intf.reset = 1'b1;
    forever #5 intf.clock = ~intf.clock;
  end

  initial begin
    uvm_config_db#(virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW))::set(null, "*", "vif", intf);
    uvm_config_db#(virtual redmule_reference_if)::set(null, "*", "reference", reference_interface);
    run_test("redmule_ball_test");
  end
endmodule
