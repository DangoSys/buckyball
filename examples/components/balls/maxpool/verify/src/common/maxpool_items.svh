class maxpool_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(maxpool_cmd_item)

  bit [31:0] input_base;
  bit [31:0] output_base;
  bit [31:0] output_stride;
  bit [31:0] input_side;
  bit [31:0] output_side;
  bit [31:0] kernel;
  bit [31:0] stride;
  bit [31:0] padding;
  bit [31:0] start_row;
  bit [31:0] start_col;
  bit [31:0] num_input_words;
  bit [31:0] num_dst_words;
  bit [127:0] input_words[MAXPOOL_MAX_INPUT_WORDS];
  bit [31:0] input_addr[MAXPOOL_MAX_INPUT_WORDS];
  bit [127:0] dst_words[MAXPOOL_MAX_DST_WORDS];
  bit [31:0] dst_addr[MAXPOOL_MAX_DST_WORDS];

  function new(string name = "maxpool_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    maxpool_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = maxpool_case_load(index);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("maxpool_case_load returned %0d for index %0d", rc, index))
    end
    maxpool_case_cmd(cmd);

    // bb_maxpool: banks and iter come from rs1, the pooling geometry from rs2.
    rob_id          = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct           = MAXPOOL_THREAD_FUNCT7[6:0];
    rs2             = {cmd.rs2_hi, cmd.rs2_lo};
    iter            = cmd.iter;
    op1_bank        = cmd.op1_bank;
    op2_bank        = '0;
    wr_bank         = cmd.wr_bank;
    op1_group       = cmd.op1_group;
    op2_group       = '0;
    wr_group        = cmd.wr_group;
    op1_col         = cmd.op1_col;
    op2_col         = '0;
    wr_col          = cmd.wr_col;
    wr_read_bank    = '0;
    wr_arch_bank    = cmd.wr_bank;
    input_base      = cmd.input_base;
    output_base     = cmd.output_base;
    output_stride   = cmd.output_stride;
    input_side      = cmd.input_side;
    output_side     = cmd.output_side;
    kernel          = cmd.kernel;
    stride          = cmd.stride;
    padding         = cmd.padding;
    start_row       = cmd.start_row;
    start_col       = cmd.start_col;
    num_input_words = cmd.num_input_words;
    num_dst_words   = cmd.num_dst_words;

    if (op1_col != 1 || wr_col != 1)
      `uvm_fatal("CASE", "MAXPOOL requires one input group and one output group")
    if (op1_bank == wr_bank) `uvm_fatal("CASE", "MAXPOOL input and output banks must differ")
    if (rs2[63:46] != 0) `uvm_fatal("CASE", "MAXPOOL reserved rs2 bits must be zero")
    if (num_dst_words != iter)
      `uvm_fatal("CASE", $sformatf("num_dst_words %0d != iter %0d", num_dst_words, iter))
    if (iter != output_side * output_side)
      `uvm_fatal("CASE", $sformatf("iter %0d != output_side^2 %0d", iter, output_side * output_side
                 ))
    if (num_input_words == 0 || num_input_words > MAXPOOL_MAX_INPUT_WORDS)
      `uvm_fatal("CASE", $sformatf("num_input_words out of range: %0d", num_input_words))
    if (num_dst_words == 0 || num_dst_words > MAXPOOL_MAX_DST_WORDS)
      `uvm_fatal("CASE", $sformatf("num_dst_words out of range: %0d", num_dst_words))

    for (i = 0; i < num_input_words; i++) begin
      w_lo = maxpool_case_input_word_lo(i);
      w_hi = maxpool_case_input_word_hi(i);
      input_words[i] = {w_hi, w_lo};
      input_addr[i] = maxpool_case_input_addr(i);
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = maxpool_case_dst_word_lo(i);
      w_hi = maxpool_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
      dst_addr[i] = maxpool_case_dst_addr(i);
    end
  endfunction

  function void do_copy(uvm_object rhs);
    maxpool_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not maxpool_cmd_item")
    end
    input_base = rhs_.input_base;
    output_base = rhs_.output_base;
    output_stride = rhs_.output_stride;
    input_side = rhs_.input_side;
    output_side = rhs_.output_side;
    kernel = rhs_.kernel;
    stride = rhs_.stride;
    padding = rhs_.padding;
    start_row = rhs_.start_row;
    start_col = rhs_.start_col;
    num_input_words = rhs_.num_input_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < MAXPOOL_MAX_INPUT_WORDS; i++) begin
      input_words[i] = rhs_.input_words[i];
      input_addr[i]  = rhs_.input_addr[i];
    end
    for (int i = 0; i < MAXPOOL_MAX_DST_WORDS; i++) begin
      dst_words[i] = rhs_.dst_words[i];
      dst_addr[i]  = rhs_.dst_addr[i];
    end
  endfunction
endclass
