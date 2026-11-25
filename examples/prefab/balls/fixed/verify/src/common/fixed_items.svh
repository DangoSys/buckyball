class fixed_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(fixed_cmd_item)
  int unsigned num_input_words, num_other_words, num_dst_words;
  bit [127:0] input_words[FIXED_MAX_ROWS];
  bit [127:0] other_words[FIXED_MAX_ROWS];
  bit [127:0] dst_words[FIXED_MAX_ROWS];
  function new(string name = "fixed_cmd_item");
    super.new(name);
  endfunction
  function void load_rust_case(int unsigned index);
    int mode = index % 6;
    int fraction = (index / 6) % 3 == 0 ? 1 : (index / 6) % 3 == 1 ? 8 : 24;
    int broadcast = mode == 3 && (index / 18) % 2;
    if (index >= 90) begin
      mode = (index - 90) % 3;
      fraction = ((index - 90) / 3) % 3 == 0 ? 1 : ((index - 90) / 3) % 3 == 1 ? 8 : 24;
      broadcast = 0;
    end else if (index >= 72) begin
      int reduction_case = (index - 72) % 9;
      mode = reduction_case < 6 ? 6 : 7;
      fraction = reduction_case % 3 == 0 ? 1 : reduction_case % 3 == 1 ? 8 : 24;
      broadcast = 0;
    end
    rob_id = index % (1 << BB_BLINK_ROB_ID_W);
    funct = FIXED_THREAD_FUNCT7;
    rs2 = mode | (broadcast << 3) | (fraction << 4) | ((index >= 90) << 9);
    iter = index < 36 ? 4 : 8;
    if (index >= 90) iter = index < 99 ? 4 : 8;
    else if (index >= 72) iter = (index - 72) % 9 < 3 || mode == 7 ? 4 : 8;
    if (iter > (1 << BB_BLINK_BANK_ADDR_W)) `uvm_fatal("CASE", "Fixed case exceeds bank depth")
    op1_bank = 1;
    op2_bank = 2;
    wr_bank = 3;
    op1_group = 0;
    op2_group = 0;
    wr_group = 0;
    op1_col = 1;
    op2_col = 1;
    wr_col = 1;
    wr_read_bank = 0;
    wr_arch_bank = wr_bank;
    num_input_words = mode == 7 ? 8 : iter;
    num_dst_words = mode == 6 ? 8 : iter;
    num_other_words = index >= 90 ? iter * 2 : mode <= 2 || mode == 6 ? 8 : mode == 7 ? 4 : mode == 4 ? 0 : broadcast ? 4 : iter;
    for (int i = 0; i < num_input_words; i++)
    input_words[i] = {fixed_case_word(index, 0, i, 1), fixed_case_word(index, 0, i, 0)};
    for (int i = 0; i < num_dst_words; i++)
    dst_words[i] = {fixed_case_word(index, 2, i, 1), fixed_case_word(index, 2, i, 0)};
    for (int i = 0; i < num_other_words; i++)
    other_words[i] = {fixed_case_word(index, 1, i, 1), fixed_case_word(index, 1, i, 0)};
  endfunction
  function void do_copy(uvm_object rhs);
    fixed_cmd_item other;
    super.do_copy(rhs);
    if (!$cast(other, rhs)) `uvm_fatal("COPY", "Expected fixed command")
    num_input_words = other.num_input_words;
    num_other_words = other.num_other_words;
    num_dst_words = other.num_dst_words;
    input_words = other.input_words;
    other_words = other.other_words;
    dst_words = other.dst_words;
  endfunction
endclass
