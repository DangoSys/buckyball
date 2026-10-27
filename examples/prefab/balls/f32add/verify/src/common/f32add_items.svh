class f32add_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(f32add_cmd_item)

  bit first;
  bit [4:0] group;
  bit [127:0] a_words[];
  bit [127:0] b_words[];
  bit [127:0] c_words[];

  function new(string name = "f32add_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    f32add_cmd_dpi_t cmd;
    int unsigned rc;

    rc = f32add_case_load(index);
    if (rc != 0)
      `uvm_fatal("CASE", $sformatf("f32add_case_load returned %0d for index %0d", rc, index))
    f32add_case_cmd(cmd);

    // bb_f32add: banks a, b, c from rs1, rows in iter; rs2[4:0] is the source
    // group of a and rs2[5] starts from +0 instead of reading b.
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = F32ADD_THREAD_FUNCT7[6:0];
    rs2          = {cmd.first[0], cmd.group[4:0]};
    iter         = cmd.rows;
    op1_bank     = cmd.a_bank;
    op2_bank     = cmd.b_bank;
    wr_bank      = cmd.c_bank;
    op1_group    = cmd.a_group_base;
    op2_group    = cmd.b_group;
    wr_group     = cmd.c_group;
    op1_col      = cmd.a_cols;
    op2_col      = 1;
    wr_col       = 1;
    wr_read_bank = '0;
    wr_arch_bank = cmd.c_bank;
    first        = cmd.first[0];
    group        = cmd.group[4:0];

    a_words      = new[cmd.rows];
    b_words      = new[cmd.rows];
    c_words      = new[cmd.rows];
    for (int i = 0; i < cmd.rows; i++) begin
      a_words[i] = {f32add_case_a_word_hi(i), f32add_case_a_word_lo(i)};
      b_words[i] = {f32add_case_b_word_hi(i), f32add_case_b_word_lo(i)};
      c_words[i] = {f32add_case_c_word_hi(i), f32add_case_c_word_lo(i)};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    f32add_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not f32add_cmd_item")
    first   = rhs_.first;
    group   = rhs_.group;
    a_words = rhs_.a_words;
    b_words = rhs_.b_words;
    c_words = rhs_.c_words;
  endfunction
endclass
