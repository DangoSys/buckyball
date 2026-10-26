class matadd_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(matadd_cmd_item)

  int unsigned lines;
  int unsigned group_count;
  int unsigned case_index;

  function new(string name = "matadd_cmd_item");
    super.new(name);
  endfunction

  // Case 0 covers one group at base 0; cases 1 and 2 cover two groups at nonzero bases.
  function void load_case(int unsigned index);
    if (index > 2) `uvm_fatal("CASE", $sformatf("unsupported case %0d", index))
    case_index = index;
    // bb_matadd: banks and iter come from rs1, rs2 is zero.
    rob_id = 4 + index;
    funct = MATADD_THREAD_FUNCT7[6:0];
    rs2 = 64'd0;
    iter = (index == 2) ? MATADD_MAX_LINES : 16;
    lines = iter;
    group_count = (index == 0) ? 1 : 2;
    op1_bank = 0;
    op2_bank = 1;
    wr_bank = 2;
    op1_group = (index == 0) ? 0 : 1;
    op2_group = (index == 0) ? 0 : 3;
    wr_group = (index == 0) ? 0 : index;
    op1_col = group_count;
    op2_col = group_count;
    wr_col = group_count;
    wr_read_bank = 0;
    wr_arch_bank = wr_bank;
  endfunction

  function automatic bit [31:0] a_lane(int unsigned group, int unsigned line, int unsigned lane);
    int unsigned offset;
    offset = (group * lines + line) * 4 + lane;
    if (case_index == 2) return 32'hffff_ff00 + offset;
    return 32'h1000_0000 + offset * 32'h0001_0011;
  endfunction

  function automatic bit [31:0] b_lane(int unsigned group, int unsigned line, int unsigned lane);
    int unsigned offset;
    offset = (group * lines + line) * 4 + lane;
    if (case_index == 2) return 32'h0000_0200 + offset * 5;
    return 32'he000_0000 + offset * 32'h0000_0103;
  endfunction

  function automatic bit [127:0] a_word(int unsigned group, int unsigned line);
    bit [127:0] word;
    for (int lane = 0; lane < 4; lane++) word[lane*32+:32] = a_lane(group, line, lane);
    return word;
  endfunction

  function automatic bit [127:0] b_word(int unsigned group, int unsigned line);
    bit [127:0] word;
    for (int lane = 0; lane < 4; lane++) word[lane*32+:32] = b_lane(group, line, lane);
    return word;
  endfunction

  function automatic bit [127:0] sum_word(int unsigned group, int unsigned line);
    bit [127:0] word;
    for (int lane = 0; lane < 4; lane++) begin
      word[lane*32+:32] = a_lane(group, line, lane) + b_lane(group, line, lane);
    end
    return word;
  endfunction

  function void do_copy(uvm_object rhs);
    matadd_cmd_item rhs_item;
    super.do_copy(rhs);
    if (!$cast(rhs_item, rhs)) `uvm_fatal("COPY", "rhs is not matadd_cmd_item")
    lines = rhs_item.lines;
    group_count = rhs_item.group_count;
    case_index = rhs_item.case_index;
  endfunction
endclass
