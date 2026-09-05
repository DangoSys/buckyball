class toint8_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(toint8_cmd_item)

  bit [31:0] kind;
  bit [31:0] input_base;
  bit [31:0] num_src_words;
  bit [31:0] num_scale_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[TOINT8_MAX_SRC_WORDS];
  bit [127:0] scale_words[TOINT8_MAX_SCALE_WORDS];
  bit [127:0] dst_words[TOINT8_MAX_DST_WORDS];
  bit [31:0] dst_addr[TOINT8_MAX_DST_WORDS];

  constraint legal_c {
    funct7 inside {QUANT_F32_TO_I8_CORE_FUNCT7[6:0], QUANT_I32_TO_I8_CORE_FUNCT7[6:0]};
    op1_en == 1'b1;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op1_col == 5'd1;
    wr_col == 5'd1;
    meta_bank == 5'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
    special == 64'd0;
  }

  function new(string name = "toint8_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    toint8_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = toint8_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("toint8_case_load returned %0d for index %0d", rc, index))
    end
    toint8_case_cmd(cmd);

    if (cmd.kind != TOINT8_KIND_F32 && cmd.kind != TOINT8_KIND_I32) begin
      `uvm_fatal("CASE", $sformatf("unknown toint8 kind %0d", cmd.kind))
    end

    kind            = cmd.kind;
    this.bid        = cmd.bid[4:0];
    iter            = cmd.iter;
    op1_bank        = cmd.op1_bank[4:0];
    op2_bank        = cmd.op2_bank[4:0];
    wr_bank         = cmd.wr_bank[4:0];
    op1_col         = cmd.op1_col[4:0];
    op2_col         = cmd.op2_col[4:0];
    wr_col          = cmd.wr_col[4:0];
    rob_id          = cmd.rob_id[3:0];
    rs1             = {cmd.rs1_hi, cmd.rs1_lo};
    rs2             = {cmd.rs2_hi, cmd.rs2_lo};
    input_base      = cmd.input_base;
    num_src_words   = cmd.num_src_words;
    num_scale_words = cmd.num_scale_words;
    num_dst_words   = cmd.num_dst_words;
    special         = 64'd0;
    op1_en          = 1'b1;
    wr_spad_en      = 1'b1;
    op1_from_spad   = 1'b1;
    meta_bank       = 5'd0;
    is_sub          = 1'b0;
    sub_rob_id      = 8'h00;

    if (kind == TOINT8_KIND_F32) begin
      funct7        = QUANT_F32_TO_I8_CORE_FUNCT7[6:0];
      op2_en        = 1'b0;
      op2_from_spad = 1'b0;
      if (op2_bank != 0 || op2_col != 0 || rs1[19:10] != 0)
        `uvm_fatal("CASE", "QUANT_F32_TO_I8 reserves input bank 1")
      if (rs2[63:32] != 0) `uvm_fatal("CASE", "QUANT_F32_TO_I8 reserves rs2[63:32]")
      if (num_scale_words != 0) `uvm_fatal("CASE", "QUANT_F32_TO_I8 must not emit scale rows")
      if (input_base != 0) `uvm_fatal("CASE", "QUANT_F32_TO_I8 input_base must be 0")
    end else begin
      funct7        = QUANT_I32_TO_I8_CORE_FUNCT7[6:0];
      op2_en        = 1'b1;
      op2_from_spad = 1'b1;
      if (op2_col != 1) `uvm_fatal("CASE", "QUANT_I32_TO_I8 scale must occupy one bank")
      if (op1_bank == op2_bank)
        `uvm_fatal("CASE", "QUANT_I32_TO_I8 input and scale banks must differ")
      if (op2_bank == wr_bank)
        `uvm_fatal("CASE", "QUANT_I32_TO_I8 scale and output banks must differ")
      if (rs2[63:35] != 0) `uvm_fatal("CASE", "QUANT_I32_TO_I8 reserves rs2[63:35]")
      if (num_scale_words != 4) `uvm_fatal("CASE", "QUANT_I32_TO_I8 must emit four scale rows")
    end

    if (iter == 0 || iter[1:0] != 2'b00)
      `uvm_fatal("CASE", "ToInt8Ball iter must be a positive multiple of four")
    if (op1_col != 1 || wr_col != 1)
      `uvm_fatal("CASE", "ToInt8Ball input and output must each occupy one bank")
    if (op1_bank == wr_bank) `uvm_fatal("CASE", "ToInt8Ball input and output banks must differ")
    if (num_src_words != iter)
      `uvm_fatal("CASE", $sformatf("num_src_words %0d != iter %0d", num_src_words, iter))
    if (num_dst_words != (iter >> 2)) `uvm_fatal("CASE", "num_dst_words does not match iter/4")
    if (num_src_words == 0 || num_src_words > TOINT8_MAX_SRC_WORDS)
      `uvm_fatal("CASE", $sformatf("num_src_words out of range: %0d", num_src_words))
    if (num_dst_words == 0 || num_dst_words > TOINT8_MAX_DST_WORDS)
      `uvm_fatal("CASE", $sformatf("num_dst_words out of range: %0d", num_dst_words))
    if (funct7 != QUANT_F32_TO_I8_CORE_FUNCT7[6:0] && funct7 != QUANT_I32_TO_I8_CORE_FUNCT7[6:0])
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))

    for (i = 0; i < num_src_words; i++) begin
      w_lo = toint8_case_src_word_lo(i);
      w_hi = toint8_case_src_word_hi(i);
      src_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_scale_words; i++) begin
      w_lo = toint8_case_scale_word_lo(i);
      w_hi = toint8_case_scale_word_hi(i);
      scale_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = toint8_case_dst_word_lo(i);
      w_hi = toint8_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
      dst_addr[i] = toint8_case_dst_addr(i);
    end
  endfunction

  function void do_copy(uvm_object rhs);
    toint8_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not toint8_cmd_item")
    end
    kind = rhs_.kind;
    input_base = rhs_.input_base;
    num_src_words = rhs_.num_src_words;
    num_scale_words = rhs_.num_scale_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < TOINT8_MAX_SRC_WORDS; i++) src_words[i] = rhs_.src_words[i];
    for (int i = 0; i < TOINT8_MAX_SCALE_WORDS; i++) scale_words[i] = rhs_.scale_words[i];
    for (int i = 0; i < TOINT8_MAX_DST_WORDS; i++) begin
      dst_words[i] = rhs_.dst_words[i];
      dst_addr[i]  = rhs_.dst_addr[i];
    end
  endfunction
endclass
