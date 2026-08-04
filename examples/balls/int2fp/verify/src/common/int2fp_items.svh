class int2fp_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(int2fp_cmd_item)

  bit [31:0] scale_bits;
  bit [31:0] output_mode;
  bit [31:0] num_src_words;
  bit [127:0] input_words[INT2FP_MAX_WORDS];

  constraint legal_c {
    funct7 == INT2FP_FUNCT7[6:0];
    op1_en == 1'b1;
    op2_en == 1'b0;
    wr_spad_en == 1'b1;
    op1_from_spad == 1'b1;
    op2_from_spad == 1'b0;
    op2_bank == 5'd0;
    op2_col == 5'd0;
    meta_bank == 5'd0;
    rs1 == 64'd0;
    rs2 == 64'd0;
    is_sub == 1'b0;
    sub_rob_id == 8'h00;
  }

  function new(string name = "int2fp_cmd_item");
    super.new(name);
  endfunction

  function bit is_i8();
    return (output_mode == INT2FP_OUTPUT_INT8) && (op1_col == 5'd4) && (wr_col == 5'd1);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index, int unsigned bid);
    int2fp_cmd_dpi_t cmd;
    int unsigned rc;
    longint unsigned w_lo;
    longint unsigned w_hi;
    int i;

    rc = int2fp_case_load(seed, index, bid);
    if (rc != 0) begin
      `uvm_fatal("CASE", $sformatf("int2fp_case_load returned %0d for index %0d", rc, index))
    end
    int2fp_case_cmd(cmd);

    bid           = cmd.bid[4:0];
    funct7        = cmd.funct7[6:0];
    iter          = cmd.iter;
    scale_bits    = cmd.scale_bits;
    output_mode   = cmd.output_mode;
    op1_bank      = cmd.op1_bank[4:0];
    wr_bank       = cmd.wr_bank[4:0];
    op1_col       = cmd.op1_col[4:0];
    wr_col        = cmd.wr_col[4:0];
    rob_id        = cmd.rob_id[3:0];
    num_src_words = cmd.num_src_words;

    // special[31:0]=scale, special[33:32]=output_mode (ISA / RTL contract)
    special       = {30'd0, output_mode[1:0], scale_bits};
    op1_en        = 1'b1;
    op2_en        = 1'b0;
    wr_spad_en    = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b0;
    op2_bank      = 5'd0;
    op2_col       = 5'd0;
    meta_bank     = 5'd0;
    rs1           = 64'd0;
    rs2           = 64'd0;
    is_sub        = 1'b0;
    sub_rob_id    = 8'h00;

    if (num_src_words == 0 || num_src_words > INT2FP_MAX_WORDS) begin
      `uvm_fatal("CASE", $sformatf("num_src_words out of range: %0d", num_src_words))
    end
    if (!((output_mode == INT2FP_OUTPUT_FP32 && op1_col == 5'd1 && wr_col == 5'd1) ||
          (output_mode == INT2FP_OUTPUT_INT8 && op1_col == 5'd4 && wr_col == 5'd1))) begin
      `uvm_fatal("CASE", $sformatf("unsupported mode/layout mode=%0d op1_col=%0d wr_col=%0d",
                                   output_mode, op1_col, wr_col))
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "op1_bank and wr_bank overlap")
    end
    if (funct7 != INT2FP_FUNCT7[6:0]) begin
      `uvm_fatal("CASE", $sformatf("invalid funct7: %0d", funct7))
    end
    if (output_mode > INT2FP_OUTPUT_INT8) begin
      `uvm_fatal("CASE", $sformatf("reserved output_mode: %0d", output_mode))
    end

    for (i = 0; i < num_src_words; i++) begin
      w_lo = int2fp_case_src_word_lo(i);
      w_hi = int2fp_case_src_word_hi(i);
      input_words[i] = {w_hi, w_lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    int2fp_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not int2fp_cmd_item")
    end
    scale_bits = rhs_.scale_bits;
    output_mode = rhs_.output_mode;
    num_src_words = rhs_.num_src_words;
    for (int i = 0; i < INT2FP_MAX_WORDS; i++) begin
      input_words[i] = rhs_.input_words[i];
    end
  endfunction
endclass
