class matrix_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(matrix_cmd_item)

  bit [31:0] m;
  bit [31:0] n;
  bit [31:0] k;
  bit [31:0] mode;
  bit [31:0] num_a_words;
  bit [31:0] num_b_words;
  bit [31:0] num_writes;
  bit [127:0] a_words[MATRIX_MAX_WORDS];
  bit [127:0] b_words[MATRIX_MAX_WORDS];

  constraint legal_c {
    funct7 == MATRIX_FUNCT7[6:0];
    op1_en == 1'b1;
    op2_en == 1'b1;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b1;
    op1_col == 5'd1;
    op2_col == 5'd1;
    wr_col == 5'd4;
    meta_bank == 5'd0;
    iter == 34'd0;
    special == 64'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
  }

  function new(string name = "matrix_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index, int unsigned bid);
    matrix_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = matrix_case_load(seed, index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("matrix_case_load returned %0d for index %0d", rc, index))
    end
    matrix_case_cmd(cmd);

    bid           = cmd.bid[4:0];
    funct7        = cmd.funct7[6:0];
    m             = cmd.m;
    n             = cmd.n;
    k             = cmd.k;
    mode          = cmd.mode;
    op1_bank      = cmd.op1_bank[4:0];
    op2_bank      = cmd.op2_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = 5'd1;
    op2_col       = 5'd1;
    wr_col        = 5'd4;
    rob_id        = cmd.rob_id[3:0];
    rs1           = {cmd.rs1_hi, cmd.rs1_lo};
    rs2           = {cmd.rs2_hi, cmd.rs2_lo};
    num_a_words   = cmd.num_a_words;
    num_b_words   = cmd.num_b_words;
    num_writes    = cmd.num_writes;

    op1_en        = 1'b1;
    op2_en        = 1'b1;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b1;
    meta_bank     = 5'd0;
    iter          = 34'd0;
    special       = 64'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (num_a_words > MATRIX_MAX_WORDS || num_b_words > MATRIX_MAX_WORDS) begin
      `uvm_fatal("CASE", $sformatf("word count out of range a=%0d b=%0d", num_a_words, num_b_words))
    end
    if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank) begin
      `uvm_fatal("CASE", "banks must be distinct")
    end
    if (op1_bank > 7 || op2_bank > 7 || wr_bank > 7) begin
      `uvm_fatal("CASE", "banks must be in 0..7")
    end

    for (i = 0; i < num_a_words; i++) begin
      w_lo = matrix_case_a_word_lo(i);
      w_hi = matrix_case_a_word_hi(i);
      a_words[i] = {w_hi, w_lo};
    end
    for (i = 0; i < num_b_words; i++) begin
      w_lo = matrix_case_b_word_lo(i);
      w_hi = matrix_case_b_word_hi(i);
      b_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    matrix_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not matrix_cmd_item")
    end
    m = rhs_.m;
    n = rhs_.n;
    k = rhs_.k;
    mode = rhs_.mode;
    num_a_words = rhs_.num_a_words;
    num_b_words = rhs_.num_b_words;
    num_writes = rhs_.num_writes;
    for (int i = 0; i < MATRIX_MAX_WORDS; i++) begin
      a_words[i] = rhs_.a_words[i];
      b_words[i] = rhs_.b_words[i];
    end
  endfunction
endclass
