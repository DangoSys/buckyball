class int8mul_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int8mul_cmd_item)

  bit [31:0] gate_row;
  bit [31:0] num_gate_words;
  bit [31:0] num_input_words;
  bit [31:0] num_dst_words;
  bit [127:0] gate_words[INT8MUL_MAX_ROWS];
  bit [127:0] input_words[INT8MUL_MAX_ROWS];
  bit [127:0] dst_words[INT8MUL_MAX_ROWS];

  constraint legal_c {
    funct7 == INT8MUL_CORE_FUNCT7[6:0];
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

  function new(string name = "int8mul_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index, int unsigned bid);
    int8mul_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int8mul_case_load(index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int8mul_case_load returned %0d for index %0d", rc, index))
    end
    int8mul_case_cmd(cmd);

    this.bid        = cmd.bid[4:0];
    iter            = cmd.iter;
    op1_bank        = cmd.gate_bank[4:0];
    op2_bank        = cmd.input_bank[4:0];
    wr_bank         = cmd.output_bank[4:0];
    op1_col         = cmd.op1_col[4:0];
    op2_col         = cmd.op2_col[4:0];
    wr_col          = cmd.wr_col[4:0];
    gate_row        = cmd.gate_row;
    rob_id          = cmd.rob_id[3:0];
    rs1             = {cmd.rs1_hi, cmd.rs1_lo};
    rs2             = {cmd.rs2_hi, cmd.rs2_lo};
    num_gate_words  = cmd.num_gate_words;
    num_input_words = cmd.num_input_words;
    num_dst_words   = cmd.num_dst_words;
    funct7          = INT8MUL_CORE_FUNCT7[6:0];
    special         = 64'd0;
    op1_en          = 1'b1;
    op2_en          = 1'b1;
    wr_spad_en      = 1'b1;
    op1_from_spad   = 1'b1;
    op2_from_spad   = 1'b1;
    meta_bank       = 5'd0;
    is_sub          = 1'b0;
    sub_rob_id      = 8'h00;

    if (iter != 1 && iter != 4)
      `uvm_fatal("CASE", $sformatf("Int8MulBall iter %0d is not in {1,4}", iter))
    if (op1_col != 1 || op2_col != 1 || wr_col != 1)
      `uvm_fatal("CASE", "Int8MulBall bank groups must match")
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
      `uvm_fatal("CASE", "Int8MulBall banks must be distinct")
    if (rs2[63:38] != 0)
      `uvm_fatal("CASE", $sformatf("Int8MulBall reserved rs2 bits set: 0x%016h", rs2))
    if (num_input_words != iter)
      `uvm_fatal("CASE", $sformatf("num_input_words %0d != iter %0d", num_input_words, iter))
    if (num_dst_words != iter)
      `uvm_fatal("CASE", $sformatf("num_dst_words %0d != iter %0d", num_dst_words, iter))
    if (funct7 != INT8MUL_CORE_FUNCT7[6:0])
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))

    for (i = 0; i < num_gate_words; i++) begin
      w_lo = int8mul_case_gate_word_lo(i);
      w_hi = int8mul_case_gate_word_hi(i);
      gate_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_input_words; i++) begin
      w_lo = int8mul_case_input_word_lo(i);
      w_hi = int8mul_case_input_word_hi(i);
      input_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_dst_words; i++) begin
      w_lo = int8mul_case_dst_word_lo(i);
      w_hi = int8mul_case_dst_word_hi(i);
      dst_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    int8mul_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not int8mul_cmd_item")
    end
    gate_row = rhs_.gate_row;
    num_gate_words = rhs_.num_gate_words;
    num_input_words = rhs_.num_input_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) gate_words[i] = rhs_.gate_words[i];
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) input_words[i] = rhs_.input_words[i];
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) dst_words[i] = rhs_.dst_words[i];
  endfunction
endclass
