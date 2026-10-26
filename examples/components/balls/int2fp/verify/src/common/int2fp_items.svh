class int2fp_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int2fp_cmd_item)

  bit [31:0] relu;
  bit [31:0] num_src_words;
  bit [31:0] num_scale_words;
  bit [31:0] num_dst_words;
  bit [127:0] src_words[INT2FP_MAX_SRC_WORDS];
  bit [127:0] scale_words[INT2FP_MAX_SCALE_WORDS];
  bit [127:0] dst_words[INT2FP_MAX_DST_WORDS];

  function new(string name = "int2fp_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    int2fp_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int2fp_case_load(index);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int2fp_case_load returned %0d for index %0d", rc, index))
    end
    int2fp_case_cmd(cmd);

    // bb_int32_to_fp32: input/scale/output banks and iter come from rs1, relu from rs2[0].
    rob_id          = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct           = INT32_TO_FP32_THREAD_FUNCT7[6:0];
    rs2             = {cmd.rs2_hi, cmd.rs2_lo};
    iter            = cmd.iter;
    op1_bank        = cmd.op1_bank;
    op2_bank        = cmd.op2_bank;
    wr_bank         = cmd.wr_bank;
    op1_group       = cmd.op1_group;
    op2_group       = cmd.op2_group;
    wr_group        = cmd.wr_group;
    op1_col         = cmd.op1_col;
    op2_col         = cmd.op2_col;
    wr_col          = cmd.wr_col;
    wr_read_bank    = '0;
    wr_arch_bank    = cmd.wr_bank;
    relu            = cmd.relu;
    num_src_words   = cmd.num_src_words;
    num_scale_words = cmd.num_scale_words;
    num_dst_words   = cmd.num_dst_words;

    if (iter == 0 || iter[1:0] != 2'b00)
      `uvm_fatal("CASE", "INT32_TO_FP32 iter must be a positive multiple of four")
    if (iter != 4 && iter != 8 && iter != 16)
      `uvm_fatal("CASE", $sformatf("INT32_TO_FP32 iter %0d is not in {4,8,16}", iter))
    if (op1_col != 1 || op2_col != 1 || wr_col != 1)
      `uvm_fatal("CASE", "INT32_TO_FP32 operands must each occupy one bank")
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
      `uvm_fatal("CASE", "INT32_TO_FP32 banks must be distinct")
    if (rs2[63:1] != 0) `uvm_fatal("CASE", "INT32_TO_FP32 reserves rs2[63:1]")
    if (rs2[0] != relu[0]) `uvm_fatal("CASE", "INT32_TO_FP32 rs2[0] must match relu")
    if (num_src_words != iter)
      `uvm_fatal("CASE", $sformatf("num_src_words %0d != iter %0d", num_src_words, iter))
    if (num_dst_words != iter)
      `uvm_fatal("CASE", $sformatf("num_dst_words %0d != iter %0d", num_dst_words, iter))
    if (num_scale_words != 4)
      `uvm_fatal("CASE", $sformatf("num_scale_words %0d, expected 4", num_scale_words))
    if (num_src_words == 0 || num_src_words > INT2FP_MAX_SRC_WORDS)
      `uvm_fatal("CASE", $sformatf("num_src_words out of range: %0d", num_src_words))
    if (num_dst_words == 0 || num_dst_words > INT2FP_MAX_DST_WORDS)
      `uvm_fatal("CASE", $sformatf("num_dst_words out of range: %0d", num_dst_words))

    for (i = 0; i < num_src_words; i++) begin
      w_lo = int2fp_case_src_word_lo(i);
      w_hi = int2fp_case_src_word_hi(i);
      src_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_scale_words; i++) begin
      w_lo = int2fp_case_scale_word_lo(i);
      w_hi = int2fp_case_scale_word_hi(i);
      scale_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = int2fp_case_dst_word_lo(i);
      w_hi = int2fp_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    int2fp_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not int2fp_cmd_item")
    end
    relu = rhs_.relu;
    num_src_words = rhs_.num_src_words;
    num_scale_words = rhs_.num_scale_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < INT2FP_MAX_SRC_WORDS; i++) src_words[i] = rhs_.src_words[i];
    for (int i = 0; i < INT2FP_MAX_SCALE_WORDS; i++) scale_words[i] = rhs_.scale_words[i];
    for (int i = 0; i < INT2FP_MAX_DST_WORDS; i++) dst_words[i] = rhs_.dst_words[i];
  endfunction
endclass
