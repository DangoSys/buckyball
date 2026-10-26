class int8mul_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int8mul_cmd_item)

  bit [31:0] gate_row;
  bit [31:0] bank_entries;
  bit [31:0] num_gate_words;
  bit [31:0] num_input_words;
  bit [31:0] num_dst_words;
  bit [127:0] gate_words[INT8MUL_MAX_ROWS];
  bit [127:0] input_words[INT8MUL_MAX_ROWS];
  bit [127:0] dst_words[INT8MUL_MAX_ROWS];

  function new(string name = "int8mul_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    int8mul_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int8mul_case_load(index);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int8mul_case_load returned %0d for index %0d", rc, index))
    end
    int8mul_case_cmd(cmd);

    // bb_int8mul: banks and iter come from rs1, ratio and gate_row from rs2.
    rob_id          = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct           = INT8MUL_THREAD_FUNCT7[6:0];
    rs2             = {cmd.rs2_hi, cmd.rs2_lo};
    iter            = cmd.iter;
    op1_bank        = cmd.gate_bank;
    op2_bank        = cmd.input_bank;
    wr_bank         = cmd.output_bank;
    op1_group       = cmd.op1_group;
    op2_group       = cmd.op2_group;
    wr_group        = cmd.wr_group;
    op1_col         = cmd.op1_col;
    op2_col         = cmd.op2_col;
    wr_col          = cmd.wr_col;
    wr_read_bank    = '0;
    wr_arch_bank    = cmd.output_bank;
    gate_row        = cmd.gate_row;
    bank_entries    = cmd.bank_entries;
    num_gate_words  = cmd.num_gate_words;
    num_input_words = cmd.num_input_words;
    num_dst_words   = cmd.num_dst_words;

    if (iter != 1 && iter != 4)
      `uvm_fatal("CASE", $sformatf("Int8MulBall iter %0d is not in {1,4}", iter))
    if (op1_col == 0 || op1_col != op2_col || op1_col != wr_col)
      `uvm_fatal("CASE", "Int8MulBall bank groups must match")
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
      `uvm_fatal("CASE", "Int8MulBall banks must be distinct")
    if (bank_entries == 0) `uvm_fatal("CASE", "Int8MulBall bank_entries must be positive")
    if (gate_row >= bank_entries)
      `uvm_fatal("CASE", $sformatf(
                 "Int8MulBall gate_row %0d exceeds bank_entries %0d", gate_row, bank_entries))
    if ((rs2 >> (32 + $clog2(bank_entries))) != 0)
      `uvm_fatal("CASE", $sformatf("Int8MulBall reserved rs2 bits set: 0x%016h", rs2))
    if (num_gate_words != op1_col)
      `uvm_fatal("CASE", $sformatf("num_gate_words %0d != col %0d", num_gate_words, op1_col))
    if (num_input_words != iter * op1_col || num_dst_words != iter * op1_col)
      `uvm_fatal(
          "CASE", $sformatf(
          "word counts %0d/%0d != iter %0d * col %0d", num_input_words, num_dst_words, iter, op1_col
          ))

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
    bank_entries = rhs_.bank_entries;
    num_gate_words = rhs_.num_gate_words;
    num_input_words = rhs_.num_input_words;
    num_dst_words = rhs_.num_dst_words;
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) gate_words[i] = rhs_.gate_words[i];
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) input_words[i] = rhs_.input_words[i];
    for (int i = 0; i < INT8MUL_MAX_ROWS; i++) dst_words[i] = rhs_.dst_words[i];
  endfunction
endclass
