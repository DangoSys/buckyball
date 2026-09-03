class smatmul_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(smatmul_cmd_item)

  bit [31:0] kind;
  bit [31:0] command_index;
  bit [31:0] command_count;
  bit [31:0] op1_words;
  bit [31:0] op2_words;

  function new(string name = "smatmul_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_command(int unsigned index);
    matrix_cmd_dpi_t cmd;
    smatmul_case_cmd(index, cmd);
    if (cmd.kind > 1) begin
      `uvm_fatal("CASE", $sformatf("invalid command kind %0d", cmd.kind))
    end

    kind = cmd.kind;
    command_index = index;
    command_count = smatmul_case_num_commands();
    bid = cmd.bid[4:0];
    rob_id = cmd.rob_id[3:0];
    op1_words = cmd.op1_words;
    op2_words = cmd.op2_words;
    rs1 = {cmd.rs1_hi, cmd.rs1_lo};
    rs2 = {cmd.rs2_hi, cmd.rs2_lo};
    op1_bank = rs1[4:0];
    op2_bank = rs1[14:10];
    wr_bank = rs1[24:20];
    iter = rs1[63:30];
    special = 64'd0;
    meta_bank = 5'd0;
    is_sub = 1'b0;
    sub_rob_id = 8'd0;

    if (kind == 0) begin
      funct7 = SMATMUL_BIAS_CORE_FUNCT7[6:0];
      op1_en = 1'b1;
      op2_en = 1'b0;
      wr_spad_en = 1'b0;
      op1_from_spad = 1'b1;
      op2_from_spad = 1'b0;
      op1_col = 5'd1;
      op2_col = 5'd0;
      wr_col = 5'd0;
      if (rs1[29:10] != 0 || iter != 4 || rs2 != 0 || op1_words != 4)
        `uvm_fatal("CASE", "invalid bias command")
    end else begin
      funct7 = SMATMUL_OS_CORE_FUNCT7[6:0];
      op1_en = 1'b1;
      op2_en = 1'b1;
      wr_spad_en = 1'b1;
      op1_from_spad = 1'b1;
      op2_from_spad = 1'b1;
      op1_col = 5'd1;
      op2_col = 5'd1;
      wr_col = 5'd1;
      if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
        `uvm_fatal("CASE", "SMATMUL_OS banks must be distinct")
      if (rs2[63:26] != 0 || rs2[23:12] != 16 || iter != 16)
        `uvm_fatal("CASE", "invalid SMATMUL_OS shape or reserved bits")
    end
  endfunction

  function void do_copy(uvm_object rhs);
    smatmul_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not smatmul_cmd_item")
    kind = rhs_.kind;
    command_index = rhs_.command_index;
    command_count = rhs_.command_count;
    op1_words = rhs_.op1_words;
    op2_words = rhs_.op2_words;
  endfunction
endclass
