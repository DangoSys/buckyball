interface fp2int_if;
  logic         clock;
  logic         reset;

  logic         cmd_req_ready;
  logic         cmd_req_valid;
  logic [  4:0] cmd_req_bits_cmd_bid;
  logic [  6:0] cmd_req_bits_cmd_funct7;
  logic [ 33:0] cmd_req_bits_cmd_iter;
  logic         cmd_req_bits_cmd_op1_en;
  logic         cmd_req_bits_cmd_op2_en;
  logic         cmd_req_bits_cmd_wr_spad_en;
  logic         cmd_req_bits_cmd_op1_from_spad;
  logic         cmd_req_bits_cmd_op2_from_spad;
  logic [ 63:0] cmd_req_bits_cmd_special;
  logic [  4:0] cmd_req_bits_cmd_op1_bank;
  logic [  4:0] cmd_req_bits_cmd_op2_bank;
  logic [  4:0] cmd_req_bits_cmd_wr_bank;
  logic [  4:0] cmd_req_bits_cmd_op1_col;
  logic [  4:0] cmd_req_bits_cmd_op2_col;
  logic [  4:0] cmd_req_bits_cmd_wr_col;
  logic [  4:0] cmd_req_bits_cmd_meta_bank;
  logic [ 63:0] cmd_req_bits_cmd_rs1;
  logic [ 63:0] cmd_req_bits_cmd_rs2;
  logic [  3:0] cmd_req_bits_rob_id;
  logic         cmd_req_bits_is_sub;
  logic [  7:0] cmd_req_bits_sub_rob_id;

  logic         cmd_resp_ready;
  logic         cmd_resp_valid;
  logic [  3:0] cmd_resp_bits_rob_id;
  logic         cmd_resp_bits_is_sub;
  logic [  7:0] cmd_resp_bits_sub_rob_id;

  logic [  4:0] bank_read_0_bank_id;
  logic [  3:0] bank_read_0_rob_id;
  logic [  4:0] bank_read_0_group_id;
  logic         bank_read_0_req_ready;
  logic         bank_read_0_req_valid;
  logic [  6:0] bank_read_0_req_bits_addr;
  logic         bank_read_0_resp_ready;
  logic         bank_read_0_resp_valid;
  logic [127:0] bank_read_0_resp_bits_data;

  logic [  4:0] bank_write_0_bank_id;
  logic [  3:0] bank_write_0_rob_id;
  logic         bank_write_0_req_ready;
  logic         bank_write_0_req_valid;
  logic [  6:0] bank_write_0_req_bits_addr;
  logic [ 15:0] bank_write_0_req_bits_mask;
  logic [127:0] bank_write_0_req_bits_data;
  logic         bank_write_0_resp_ready;
  logic         bank_write_0_resp_valid;
  logic         bank_write_0_resp_bits_ok;

  logic         sub_rob_req_ready;

  logic         mmio_read_req_ready;
  logic         mmio_read_resp_valid;
  logic [  7:0] mmio_read_resp_bits_data;
endinterface
