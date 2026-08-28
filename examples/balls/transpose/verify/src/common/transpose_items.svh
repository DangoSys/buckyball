class transpose_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(transpose_cmd_item)

  bit [31:0] elem_bits;
  bit [31:0] num_src_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[TRANSPOSE_MAX_WORDS];
  bit [127:0] dst_words[TRANSPOSE_MAX_WORDS];

  constraint legal_c {
    funct7 == TRANSPOSE_CORE_FUNCT7[6:0];
    op1_en == 1'b1;
    op2_en == 1'b0;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b0;
    op2_bank == 5'd0;
    op2_col == 5'd0;
    meta_bank == 5'd0;
    rs1 == 64'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
    special == 64'd0;
  }

  function new(string name = "transpose_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index, int unsigned bid);
    transpose_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = transpose_case_load(seed, index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("transpose_case_load returned %0d for index %0d", rc, index))
    end
    transpose_case_cmd(cmd);

    bid           = cmd.bid[4:0];
    funct7        = TRANSPOSE_CORE_FUNCT7[6:0];
    iter          = cmd.iter;
    op1_bank      = cmd.op1_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    elem_bits     = cmd.elem_bits;
    num_src_words = cmd.num_src_words;
    num_dst_words = cmd.num_dst_words;

    rs2           = {56'd0, elem_bits[7:0]};
    op1_en        = 1'b1;
    op2_en        = 1'b0;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b0;
    op2_bank      = 5'd0;
    op2_col       = 5'd0;
    meta_bank     = 5'd0;
    rs1           = 64'd0;
    special       = 64'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (num_src_words > TRANSPOSE_MAX_WORDS || num_dst_words > TRANSPOSE_MAX_WORDS) begin
      `uvm_fatal("CASE", $sformatf("word count out of range src=%0d dst=%0d", num_src_words,
                                   num_dst_words))
    end
    if (num_src_words != num_dst_words) begin
      `uvm_fatal("CASE", $sformatf("src/dst word count mismatch src=%0d dst=%0d", num_src_words,
                                   num_dst_words))
    end
    if (!(elem_bits == 32'd8 || elem_bits == 32'd32)) begin
      `uvm_fatal("CASE", $sformatf("unsupported elem_bits=%0d", elem_bits))
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "op1_bank and wr_bank overlap")
    end

    for (i = 0; i < num_src_words; i++) begin
      w_lo = transpose_case_src_word_lo(i);
      w_hi = transpose_case_src_word_hi(i);
      src_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = transpose_case_dst_word_lo(i);
      w_hi = transpose_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    transpose_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not transpose_cmd_item")
    end
    elem_bits = rhs_.elem_bits;
    num_src_words = rhs_.num_src_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < TRANSPOSE_MAX_WORDS; i++) begin
      src_words[i] = rhs_.src_words[i];
      dst_words[i] = rhs_.dst_words[i];
    end
  endfunction
endclass
