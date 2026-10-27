class vector_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(vector_cmd_item)

  bit [127:0] at_words[];
  bit [127:0] b_words[];
  // Indexed group * VECMAT16_ROWS + row.
  bit [127:0] c_old_words[VECMAT16_GROUPS * VECMAT16_ROWS];
  bit [127:0] c_new_words[VECMAT16_GROUPS * VECMAT16_ROWS];

  function new(string name = "vector_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    vector_cmd_dpi_t cmd;
    int unsigned rc;

    rc = vector_case_load(index);
    if (rc != 0)
      `uvm_fatal("CASE", $sformatf("vector_case_load returned %0d for index %0d", rc, index))
    vector_case_cmd(cmd);

    // bb_vecmat16: A^T, B and C from rs1 banks 0..2, K in iter; the old C
    // arrives as the renamed wr_read_bank at the wr group base. rs2 is
    // reserved and zero.
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = VECMAT16_THREAD_FUNCT7[6:0];
    rs2          = '0;
    iter         = cmd.iter;
    op1_bank     = cmd.op1_bank;
    op2_bank     = cmd.op2_bank;
    wr_bank      = cmd.wr_bank;
    op1_group    = cmd.op1_group;
    op2_group    = cmd.op2_group;
    wr_group     = cmd.wr_group;
    op1_col      = 1;
    op2_col      = 1;
    wr_col       = VECMAT16_GROUPS;
    wr_read_bank = cmd.wr_read_bank;
    wr_arch_bank = cmd.wr_bank;

    at_words     = new[cmd.iter];
    b_words      = new[cmd.iter];
    for (int i = 0; i < cmd.iter; i++) begin
      at_words[i] = {vector_case_at_word_hi(i), vector_case_at_word_lo(i)};
      b_words[i]  = {vector_case_b_word_hi(i), vector_case_b_word_lo(i)};
    end
    for (int i = 0; i < VECMAT16_GROUPS * VECMAT16_ROWS; i++) begin
      c_old_words[i] = {vector_case_c_old_word_hi(i), vector_case_c_old_word_lo(i)};
      c_new_words[i] = {vector_case_c_new_word_hi(i), vector_case_c_new_word_lo(i)};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    vector_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not vector_cmd_item")
    at_words    = rhs_.at_words;
    b_words     = rhs_.b_words;
    c_old_words = rhs_.c_old_words;
    c_new_words = rhs_.c_new_words;
  endfunction
endclass
