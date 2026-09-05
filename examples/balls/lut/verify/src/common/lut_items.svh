class lut_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(lut_cmd_item)

  bit [31:0] num_src_words;
  bit [31:0] num_lut_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[LUT_MAX_SRC_WORDS];
  bit [127:0] lut_words[LUT_LANE_TABLE_ROWS];
  bit [127:0] dst_words[LUT_MAX_DST_WORDS];

  constraint legal_c {
    funct7 == LUT_CORE_FUNCT7[6:0];
    op1_en == 1'b1;
    op2_en == 1'b1;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b1;
    op1_col == 5'd1;
    op2_col inside {5'd1, 5'd4};
    wr_col == 5'd1;
    meta_bank == 5'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
    special == 64'd0;
    rs2 == 64'd0;
  }

  function new(string name = "lut_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    lut_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = lut_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("lut_case_load returned %0d for index %0d", rc, index))
    end
    lut_case_cmd(cmd);

    this.bid      = cmd.bid[4:0];
    iter          = cmd.iter;
    op1_bank      = cmd.op1_bank[4:0];
    op2_bank      = cmd.op2_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    op2_col       = cmd.op2_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    rs1           = {cmd.rs1_hi, cmd.rs1_lo};
    rs2           = {cmd.rs2_hi, cmd.rs2_lo};
    num_src_words = cmd.num_src_words;
    num_lut_words = cmd.num_lut_words;
    num_dst_words = cmd.num_dst_words;
    funct7        = LUT_CORE_FUNCT7[6:0];
    special       = 64'd0;
    op1_en        = 1'b1;
    op2_en        = 1'b1;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b1;
    meta_bank     = 5'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (iter != 1 && iter != 4 && iter != 8)
      `uvm_fatal("CASE", $sformatf("LUT iter %0d is not in {1,4,8}", iter))
    if (op1_col != 1 || (op2_col != 1 && op2_col != 4) || wr_col != 1)
      `uvm_fatal("CASE", "LUT requires col=1 input/output and col=1 or col=4 table")
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
      `uvm_fatal("CASE", "LUT banks must be distinct")
    if (rs2 != 64'd0) `uvm_fatal("CASE", "LUT rs2 must be zero")
    if (num_src_words != iter)
      `uvm_fatal("CASE", $sformatf("num_src_words %0d != iter %0d", num_src_words, iter))
    if (num_dst_words != iter)
      `uvm_fatal("CASE", $sformatf("num_dst_words %0d != iter %0d", num_dst_words, iter))
    if (num_lut_words != (op2_col == 4 ? LUT_LANE_TABLE_ROWS : LUT_SHARED_TABLE_ROWS))
      `uvm_fatal("CASE", $sformatf(
                 "num_lut_words %0d does not match col %0d", num_lut_words, op2_col))

    for (i = 0; i < num_src_words; i++) begin
      w_lo = lut_case_src_word_lo(i);
      w_hi = lut_case_src_word_hi(i);
      src_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_lut_words; i++) begin
      w_lo = lut_case_lut_word_lo(i);
      w_hi = lut_case_lut_word_hi(i);
      lut_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = lut_case_dst_word_lo(i);
      w_hi = lut_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    lut_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not lut_cmd_item")
    end
    num_src_words = rhs_.num_src_words;
    num_lut_words = rhs_.num_lut_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < LUT_MAX_SRC_WORDS; i++) src_words[i] = rhs_.src_words[i];
    for (int i = 0; i < LUT_LANE_TABLE_ROWS; i++) lut_words[i] = rhs_.lut_words[i];
    for (int i = 0; i < LUT_MAX_DST_WORDS; i++) dst_words[i] = rhs_.dst_words[i];
  endfunction
endclass
