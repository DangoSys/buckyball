class fp2int_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(fp2int_cmd_item)

  bit [31:0] da_bits;
  bit [12:0] da_addr;
  bit [31:0] num_src_words;
  bit [127:0] input_words[FP2INT_MAX_SOURCE_WORDS];

  constraint legal_c {
    funct7 == FP2INT_CORE_FUNCT7[6:0];
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

  function new(string name = "fp2int_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    fp2int_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = fp2int_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("fp2int_case_load returned %0d for index %0d", rc, index))
    end
    fp2int_case_cmd(cmd);

    this.bid      = cmd.bid[4:0];
    funct7        = FP2INT_CORE_FUNCT7[6:0];
    iter          = cmd.iter;
    da_bits       = cmd.da_bits;
    da_addr       = 13'd0;
    op1_bank      = cmd.op1_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    num_src_words = cmd.num_src_words;

    special       = {51'd0, da_addr};
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

    if (num_src_words == 0 || num_src_words > FP2INT_MAX_SOURCE_WORDS) begin
      `uvm_fatal("CASE", $sformatf("num_src_words out of range: %0d", num_src_words))
    end
    if (op1_col == 0 || wr_col == 0 || num_src_words != iter * op1_col) begin
      `uvm_fatal("CASE", "invalid source stream")
    end
    if (num_src_words % (FP2INT_NUM_WORDS * wr_col) != 0) begin
      `uvm_fatal("CASE", "source stream does not fill destination rows")
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "op1_bank and wr_bank overlap")
    end
    if (funct7 != FP2INT_CORE_FUNCT7[6:0]) begin
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))
    end

    for (i = 0; i < num_src_words; i++) begin
      w_lo = fp2int_case_src_word_lo(i);
      w_hi = fp2int_case_src_word_hi(i);
      input_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    fp2int_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not fp2int_cmd_item")
    end
    da_bits = rhs_.da_bits;
    da_addr = rhs_.da_addr;
    num_src_words = rhs_.num_src_words;
    for (int i = 0; i < FP2INT_MAX_SOURCE_WORDS; i++) begin
      input_words[i] = rhs_.input_words[i];
    end
  endfunction
endclass
