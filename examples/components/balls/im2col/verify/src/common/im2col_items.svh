class im2col_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(im2col_cmd_item)

  bit [31:0] ksize;
  bit [31:0] stride;
  bit [31:0] padding;
  bit [31:0] window_count;
  bit [31:0] num_src_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[IM2COL_MAX_WORDS];
  bit [127:0] dst_words[IM2COL_MAX_WORDS];

  function new(string name = "im2col_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index);
    im2col_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = im2col_case_load(seed, index);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("im2col_case_load returned %0d for index %0d", rc, index))
    end
    im2col_case_cmd(cmd);

    ksize = cmd.ksize;
    stride = cmd.stride;
    padding = cmd.padding;
    window_count = cmd.window_count;
    num_src_words = cmd.num_src_words;
    num_dst_words = cmd.num_dst_words;

    // bb_im2col: banks and input_size (iter) come from rs1, the window geometry from rs2.
    rob_id = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct = IM2COL_THREAD_FUNCT7[6:0];
    rs2 = {
      1'b0, window_count[6:0], 6'd0, 4'd0, 6'd0, 8'd0, 8'd0, padding[7:0], stride[7:0], ksize[7:0]
    };
    iter = cmd.iter;
    op1_bank = cmd.op1_bank;
    op2_bank = '0;
    wr_bank = cmd.wr_bank;
    op1_group = cmd.op1_group;
    op2_group = '0;
    wr_group = cmd.wr_group;
    op1_col = cmd.op1_col;
    op2_col = '0;
    wr_col = cmd.wr_col;
    wr_read_bank = '0;
    wr_arch_bank = cmd.wr_bank;

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
    window_count = rhs_.window_count;
    num_src_words = rhs_.num_src_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < IM2COL_MAX_WORDS; i++) begin
      src_words[i] = rhs_.src_words[i];
      dst_words[i] = rhs_.dst_words[i];
    end
  endfunction
endclass
