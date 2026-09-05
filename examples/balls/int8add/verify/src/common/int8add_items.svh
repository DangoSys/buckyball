class int8add_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int8add_cmd_item)

  bit [31:0] relu;
  bit [31:0] num_lhs_words;
  bit [31:0] num_rhs_words;
  bit [31:0] num_dst_words;
  bit [127:0] lhs_words[INT8ADD_MAX_ROWS];
  bit [127:0] rhs_words[INT8ADD_MAX_ROWS];
  bit [127:0] dst_words[INT8ADD_MAX_ROWS];

  constraint legal_c {
    funct7 inside {INT8ADD_CORE_FUNCT7[6:0], INT8ADD_RELU_CORE_FUNCT7[6:0]};
    op1_en == 1'b1;
    op2_en == 1'b1;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b1;
    op1_col == 5'd1;
    op2_col == 5'd1;
    wr_col == 5'd1;
    meta_bank == 5'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
    special == 64'd0;
  }

  function new(string name = "int8add_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    int8add_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int8add_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int8add_case_load returned %0d for index %0d", rc, index))
    end
    int8add_case_cmd(cmd);

    if (cmd.relu != 0 && cmd.relu != 1) begin
      `uvm_fatal("CASE", $sformatf("invalid relu flag %0d", cmd.relu))
    end

    relu          = cmd.relu;
    this.bid      = cmd.bid[4:0];
    iter          = cmd.iter;
    op1_bank      = cmd.op1_bank[4:0];
    op2_bank      = cmd.op2_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    op2_col       = cmd.op2_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    rs1           = {cmd.rs1_hi, cmd.rs1_lo};
    rs2           = {cmd.rs2_hi, cmd.rs2_lo};
    num_lhs_words = cmd.num_lhs_words;
    num_rhs_words = cmd.num_rhs_words;
    num_dst_words = cmd.num_dst_words;
    special       = 64'd0;
    op1_en        = 1'b1;
    op2_en        = 1'b1;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b1;
    meta_bank     = 5'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (relu == 0) funct7 = INT8ADD_CORE_FUNCT7[6:0];
    else funct7 = INT8ADD_RELU_CORE_FUNCT7[6:0];

    if (iter == 0) `uvm_fatal("CASE", "Int8AddBall iter must be positive")
    if (op1_col != 1 || op2_col != 1 || wr_col != 1)
      `uvm_fatal("CASE", "Int8AddBall bank groups must match")
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
      `uvm_fatal("CASE", "Int8AddBall banks must be distinct")
    if (num_lhs_words != iter)
      `uvm_fatal("CASE", $sformatf("num_lhs_words %0d != iter %0d", num_lhs_words, iter))
    if (num_rhs_words != iter)
      `uvm_fatal("CASE", $sformatf("num_rhs_words %0d != iter %0d", num_rhs_words, iter))
    if (num_dst_words != iter)
      `uvm_fatal("CASE", $sformatf("num_dst_words %0d != iter %0d", num_dst_words, iter))
    if (funct7 != INT8ADD_CORE_FUNCT7[6:0] && funct7 != INT8ADD_RELU_CORE_FUNCT7[6:0])
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))

    for (i = 0; i < num_lhs_words; i++) begin
      w_lo = int8add_case_lhs_word_lo(i);
      w_hi = int8add_case_lhs_word_hi(i);
      lhs_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_rhs_words; i++) begin
      w_lo = int8add_case_rhs_word_lo(i);
      w_hi = int8add_case_rhs_word_hi(i);
      rhs_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = int8add_case_dst_word_lo(i);
      w_hi = int8add_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    int8add_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not int8add_cmd_item")
    end
    relu = rhs_.relu;
    num_lhs_words = rhs_.num_lhs_words;
    num_rhs_words = rhs_.num_rhs_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < INT8ADD_MAX_ROWS; i++) lhs_words[i] = rhs_.lhs_words[i];
    for (int i = 0; i < INT8ADD_MAX_ROWS; i++) rhs_words[i] = rhs_.rhs_words[i];
    for (int i = 0; i < INT8ADD_MAX_ROWS; i++) dst_words[i] = rhs_.dst_words[i];
  endfunction
endclass
