class relu_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(relu_cmd_item)

  bit [31:0] num_words;
  bit [31:0] stride;
  bit [31:0] rows[];
  bit [127:0] src_words[];
  bit [127:0] dst_words[];

  function new(string name = "relu_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    relu_cmd_dpi_t cmd;
    int unsigned   rc;

    rc = relu_case_load(index);
    if (rc != 0)
      `uvm_fatal("CASE", $sformatf("relu_case_load returned %0d for index %0d", rc, index))
    relu_case_cmd(cmd);

    // bb_relu: one bank is both source and destination; rs1 bank1 names the
    // group inside it, which arrives in the op2 bank field. rs2 is the stride.
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = RELU_THREAD_FUNCT7[6:0];
    rs2          = cmd.stride;
    iter         = cmd.iter;
    op1_bank     = cmd.bank;
    op2_bank     = cmd.group;
    wr_bank      = cmd.bank;
    op1_group    = cmd.group_base;
    op2_group    = '0;
    wr_group     = cmd.group_base;
    op1_col      = cmd.cols;
    op2_col      = '0;
    wr_col       = cmd.cols;
    wr_read_bank = '0;
    wr_arch_bank = cmd.bank;
    stride       = cmd.stride;
    num_words    = cmd.num_words;

    if (num_words == 0) `uvm_fatal("CASE", "ReluBall case has no rows")
    rows      = new[num_words];
    src_words = new[num_words];
    dst_words = new[num_words];
    for (int i = 0; i < num_words; i++) begin
      rows[i]      = relu_case_row(i);
      src_words[i] = {relu_case_src_word_hi(i), relu_case_src_word_lo(i)};
      dst_words[i] = {relu_case_dst_word_hi(i), relu_case_dst_word_lo(i)};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    relu_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not relu_cmd_item")
    num_words = rhs_.num_words;
    stride    = rhs_.stride;
    rows      = rhs_.rows;
    src_words = rhs_.src_words;
    dst_words = rhs_.dst_words;
  endfunction
endclass
