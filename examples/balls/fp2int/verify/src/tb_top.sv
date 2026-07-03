module tb_top;
  import uvm_pkg::*;
  import fp2int_pkg::*;

  fp2int_if intf ();

  Fp2IntBall dut (
      .clock(intf.clock),
      .reset(intf.reset),
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
      .io_bankRead_0_bank_id(intf.bank_read_0_bank_id),
      .io_bankRead_0_rob_id(intf.bank_read_0_rob_id),
      .io_bankRead_0_group_id(intf.bank_read_0_group_id),
      .io_bankRead_0_io_req_ready(intf.bank_read_0_req_ready),
      .io_bankRead_0_io_req_valid(intf.bank_read_0_req_valid),
      .io_bankRead_0_io_req_bits_addr(intf.bank_read_0_req_bits_addr),
      .io_bankRead_0_io_resp_ready(intf.bank_read_0_resp_ready),
      .io_bankRead_0_io_resp_valid(intf.bank_read_0_resp_valid),
      .io_bankRead_0_io_resp_bits_data(intf.bank_read_0_resp_bits_data),
      .io_bankWrite_0_bank_id(intf.bank_write_0_bank_id),
      .io_bankWrite_0_rob_id(intf.bank_write_0_rob_id),
      .io_bankWrite_0_io_req_ready(intf.bank_write_0_req_ready),
      .io_bankWrite_0_io_req_valid(intf.bank_write_0_req_valid),
      .io_bankWrite_0_io_req_bits_addr(intf.bank_write_0_req_bits_addr),
      .io_bankWrite_0_io_req_bits_mask_0(intf.bank_write_0_req_bits_mask[0]),
      .io_bankWrite_0_io_req_bits_mask_1(intf.bank_write_0_req_bits_mask[1]),
      .io_bankWrite_0_io_req_bits_mask_2(intf.bank_write_0_req_bits_mask[2]),
      .io_bankWrite_0_io_req_bits_mask_3(intf.bank_write_0_req_bits_mask[3]),
      .io_bankWrite_0_io_req_bits_mask_4(intf.bank_write_0_req_bits_mask[4]),
      .io_bankWrite_0_io_req_bits_mask_5(intf.bank_write_0_req_bits_mask[5]),
      .io_bankWrite_0_io_req_bits_mask_6(intf.bank_write_0_req_bits_mask[6]),
      .io_bankWrite_0_io_req_bits_mask_7(intf.bank_write_0_req_bits_mask[7]),
      .io_bankWrite_0_io_req_bits_mask_8(intf.bank_write_0_req_bits_mask[8]),
      .io_bankWrite_0_io_req_bits_mask_9(intf.bank_write_0_req_bits_mask[9]),
      .io_bankWrite_0_io_req_bits_mask_10(intf.bank_write_0_req_bits_mask[10]),
      .io_bankWrite_0_io_req_bits_mask_11(intf.bank_write_0_req_bits_mask[11]),
      .io_bankWrite_0_io_req_bits_mask_12(intf.bank_write_0_req_bits_mask[12]),
      .io_bankWrite_0_io_req_bits_mask_13(intf.bank_write_0_req_bits_mask[13]),
      .io_bankWrite_0_io_req_bits_mask_14(intf.bank_write_0_req_bits_mask[14]),
      .io_bankWrite_0_io_req_bits_mask_15(intf.bank_write_0_req_bits_mask[15]),
      .io_bankWrite_0_io_req_bits_data(intf.bank_write_0_req_bits_data),
      .io_bankWrite_0_io_resp_ready(intf.bank_write_0_resp_ready),
      .io_bankWrite_0_io_resp_valid(intf.bank_write_0_resp_valid),
      .io_bankWrite_0_io_resp_bits_ok(intf.bank_write_0_resp_bits_ok),
      .io_subRobReq_ready(intf.sub_rob_req_ready),
      .io_mmioRead_req_ready(intf.mmio_read_req_ready),
      .io_mmioRead_resp_valid(intf.mmio_read_resp_valid),
      .io_mmioRead_resp_bits_data(intf.mmio_read_resp_bits_data)
  );

  initial begin
    intf.clock = 1'b0;
    forever #5 intf.clock = ~intf.clock;
  end

  initial begin
    uvm_config_db#(virtual fp2int_if)::set(null, "uvm_test_top", "vif", intf);
    run_test("fp2int_ball_test");
  end
endmodule
