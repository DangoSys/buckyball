module tb_top;
  import uvm_pkg::*;
  import mxint8quantize_pkg::*;

  bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) intf ();

  Mxint8quantizeBall dut (
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
      .blink_bankWrite_0_bank_id(intf.bank_write_bank_id[0]),
      .blink_bankWrite_0_rob_id(intf.bank_write_rob_id[0]),
      .blink_bankWrite_0_group_id(intf.bank_write_group_id[0]),
      .blink_bankWrite_0_io_req_ready(intf.bank_write_req_ready[0]),
      .blink_bankWrite_0_io_req_valid(intf.bank_write_req_valid[0]),
      .blink_bankWrite_0_io_req_bits_addr(intf.bank_write_req_addr[0]),
      .blink_bankWrite_0_io_req_bits_data(intf.bank_write_req_data[0]),
      .blink_bankWrite_0_io_resp_ready(intf.bank_write_resp_ready[0]),
      .blink_bankWrite_0_io_resp_valid(intf.bank_write_resp_valid[0]),
      .blink_bankWrite_0_io_resp_bits_ok(intf.bank_write_resp_ok[0])
  );

  // Mxint8quantizeBall writes every bank row whole (mask all true in the
  // Scala source); firtool folds the constant mask out of the port list.
  assign intf.bank_write_req_mask[0] = 16'hffff;

  initial begin
    intf.clock = 1'b0;
    intf.reset = 1'b1;
    forever #5 intf.clock = ~intf.clock;
  end

  initial begin
    uvm_config_db#(virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW))::set(null, "*", "vif", intf);
    run_test("mxint8quantize_ball_test");
  end
endmodule
