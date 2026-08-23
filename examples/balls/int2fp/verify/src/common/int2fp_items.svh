class int2fp_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int2fp_cmd_item)

  bit [ 31:0] da_bits;
  bit [ 31:0] dw_bits       [INT2FP_NUM_GROUPS * 4];
  bit [ 12:0] da_addr;
  bit [ 12:0] dw_addr;
  bit         per_channel;
  bit [ 31:0] num_src_words;
  bit [127:0] input_words   [     INT2FP_MAX_WORDS];

  constraint legal_c {
    funct7 inside {INT2FP_TENSOR_CORE_FUNCT7[6:0], INT2FP_CHANNEL_CORE_FUNCT7[6:0]};
    op1_en == 1'b1;
    op2_en == 1'b0;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b0;
    op2_bank == 5'd0;
    op2_col == 5'd0;
    meta_bank == 5'd0;
    rs1 == 64'd0;
    rs2 == 64'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
  }

  function new(string name = "int2fp_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    int2fp_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int2fp_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int2fp_case_load returned %0d for index %0d", rc, index))
    end
    int2fp_case_cmd(cmd);

    this.bid      = cmd.bid[4:0];
    iter          = cmd.iter;
    da_bits       = cmd.da_bits;
    da_addr       = 13'd0;
    dw_addr       = cmd.dw_addr[12:0];
    per_channel   = cmd.per_channel[0];
    funct7        = per_channel ? INT2FP_CHANNEL_CORE_FUNCT7[6:0] : INT2FP_TENSOR_CORE_FUNCT7[6:0];
    op1_bank      = cmd.op1_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    num_src_words = cmd.num_src_words;

    special       = {38'd0, dw_addr, da_addr};
    op1_en        = 1'b1;
    op2_en        = 1'b0;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b0;
    op2_bank      = 5'd0;
    op2_col       = 5'd0;
    meta_bank     = 5'd0;
    rs1           = 64'd0;
    rs2           = 64'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (num_src_words == 0 || num_src_words > INT2FP_MAX_WORDS) begin
      `uvm_fatal("CASE", $sformatf("num_src_words out of range: %0d", num_src_words))
    end
    if (op1_col == 0 || op1_col > INT2FP_NUM_GROUPS || wr_col != op1_col) begin
      `uvm_fatal("CASE", $sformatf("unsupported layout op1_col=%0d wr_col=%0d", op1_col, wr_col))
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "op1_bank and wr_bank overlap")
    end
    if (dw_addr < 13'd16 || dw_addr[1:0] != 2'd0) begin
      `uvm_fatal("CASE", $sformatf("invalid Dw address: %0d", dw_addr))
    end
    if (funct7 != (per_channel ? INT2FP_CHANNEL_CORE_FUNCT7[6:0]
                               : INT2FP_TENSOR_CORE_FUNCT7[6:0])) begin
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))
    end
    for (i = 0; i < INT2FP_NUM_GROUPS * 4; i++) begin
      dw_bits[i] = int2fp_case_dw_bits(i);
    end
    for (i = 0; i < num_src_words; i++) begin
      w_lo = int2fp_case_src_word_lo(i);
      w_hi = int2fp_case_src_word_hi(i);
      input_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    int2fp_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not int2fp_cmd_item")
    end
    da_bits = rhs_.da_bits;
    for (int i = 0; i < INT2FP_NUM_GROUPS * 4; i++) begin
      dw_bits[i] = rhs_.dw_bits[i];
    end
    da_addr = rhs_.da_addr;
    dw_addr = rhs_.dw_addr;
    per_channel = rhs_.per_channel;
    num_src_words = rhs_.num_src_words;
    for (int i = 0; i < INT2FP_MAX_WORDS; i++) begin
      input_words[i] = rhs_.input_words[i];
    end
  endfunction
endclass
