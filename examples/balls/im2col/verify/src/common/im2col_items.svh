class im2col_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(im2col_cmd_item)

  bit [31:0] ksize;
  bit [31:0] stride;
  bit [31:0] padding;
  bit [31:0] num_src_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[IM2COL_MAX_WORDS];
  bit [127:0] dst_words[IM2COL_MAX_WORDS];

  constraint legal_c {
    funct7 == IM2COL_FUNCT7[6:0];
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

  function new(string name = "im2col_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index, int unsigned bid);
    im2col_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = im2col_case_load(seed, index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("im2col_case_load returned %0d for index %0d", rc, index))
    end
    im2col_case_cmd(cmd);

    bid           = cmd.bid[4:0];
    funct7        = cmd.funct7[6:0];
    iter          = cmd.iter;
    ksize         = cmd.ksize;
    stride        = cmd.stride;
    padding       = cmd.padding;
    op1_bank      = cmd.op1_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    num_src_words = cmd.num_src_words;
    num_dst_words = cmd.num_dst_words;

    special       = {40'd0, padding[7:0], stride[7:0], ksize[7:0]};
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

    if (num_src_words > IM2COL_MAX_WORDS || num_dst_words > IM2COL_MAX_WORDS) begin
      `uvm_fatal("CASE", $sformatf("word count out of range src=%0d dst=%0d", num_src_words,
                                   num_dst_words))
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "op1_bank and wr_bank overlap")
    end
    if (stride == 0) begin
      `uvm_fatal("CASE", "stride must be >= 1")
    end

    for (i = 0; i < num_src_words; i++) begin
      w_lo = im2col_case_src_word_lo(i);
      w_hi = im2col_case_src_word_hi(i);
      src_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = im2col_case_dst_word_lo(i);
      w_hi = im2col_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    im2col_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not im2col_cmd_item")
    end
    ksize = rhs_.ksize;
    stride = rhs_.stride;
    padding = rhs_.padding;
    num_src_words = rhs_.num_src_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < IM2COL_MAX_WORDS; i++) begin
      src_words[i] = rhs_.src_words[i];
      dst_words[i] = rhs_.dst_words[i];
    end
  endfunction
endclass
