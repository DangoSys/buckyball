class fp2int_cmd_item extends bb_blink_cmd_item;
  bit [127:0] input_words[FP2INT_NUM_WORDS];

  constraint legal_c {
    bid == 5'd5;
    funct7 == 7'd51;
    iter inside {[1 : FP2INT_NUM_WORDS]};
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

  constraint i32_layout_c {
    op1_col == 5'd1;
    wr_col == 5'd1;
  }

  `uvm_object_utils(fp2int_cmd_item)

  function new(string name = "fp2int_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned seed, int unsigned index);
    int unsigned v;
    longint unsigned word_lo;
    longint unsigned word_hi;

    v = fp2int_case_bid(seed, index);
    bid = v[4:0];
    v = fp2int_case_funct7(seed, index);
    funct7 = v[6:0];
    iter = fp2int_case_iter(seed, index);
    op1_en = 1'b1;
    op2_en = 1'b0;
    wr_spad_en = 1'b1;
    op1_from_spad = 1'b1;
    op2_from_spad = 1'b0;
    scale_bits = fp2int_case_scale_bits(seed, index);
    v = fp2int_case_op1_bank(seed, index);
    op1_bank = v[4:0];
    v = fp2int_case_op2_bank(seed, index);
    op2_bank = v[4:0];
    v = fp2int_case_wr_bank(seed, index);
    wr_bank = v[4:0];
    v = fp2int_case_op1_col(seed, index);
    op1_col = v[4:0];
    v = fp2int_case_op2_col(seed, index);
    op2_col = v[4:0];
    v = fp2int_case_wr_col(seed, index);
    wr_col = v[4:0];
    v = fp2int_case_meta_bank(seed, index);
    meta_bank = v[4:0];
    rs1 = 64'd0;
    rs2 = 64'd0;
    v = fp2int_case_rob_id(seed, index);
    rob_id = v[3:0];
    v = fp2int_case_is_sub(seed, index);
    is_sub = v[0];
    v = fp2int_case_sub_rob_id(seed, index);
    sub_rob_id = v[7:0];

    for (int i = 0; i < FP2INT_NUM_WORDS; i++) begin
      word_hi = fp2int_case_word_hi(seed, index, i);
      word_lo = fp2int_case_word_lo(seed, index, i);
      input_words[i] = {word_hi, word_lo};
    end

    check_legal();
  endfunction

  function void check_legal();
    if (bid != 5'd5) begin
      `uvm_fatal("CASE", $sformatf("invalid bid from Rust casegen: %0d", bid))
    end
    if (funct7 != 7'd51) begin
      `uvm_fatal("CASE", $sformatf("invalid funct7 from Rust casegen: %0d", funct7))
    end
    if (iter == 0 || iter > FP2INT_NUM_WORDS) begin
      `uvm_fatal("CASE", $sformatf("invalid iter from Rust casegen: %0d", iter))
    end
    if (op1_col != 5'd1 || wr_col != 5'd1) begin
      `uvm_fatal("CASE", "Rust casegen produced unsupported layout")
    end
    if (op1_bank == wr_bank) begin
      `uvm_fatal("CASE", "Rust casegen produced overlapping read/write bank")
    end
  endfunction

  function void do_copy(uvm_object rhs);
    fp2int_cmd_item rhs_;

    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) begin
      `uvm_fatal("COPY", "rhs is not fp2int_cmd_item")
    end

    for (int i = 0; i < FP2INT_NUM_WORDS; i++) begin
      input_words[i] = rhs_.input_words[i];
    end
  endfunction
endclass

class fp2int_read_item extends bb_blink_read_item;
  `uvm_object_utils(fp2int_read_item)

  function new(string name = "fp2int_read_item");
    super.new(name);
  endfunction
endclass

class fp2int_write_item extends bb_blink_write_item;
  `uvm_object_utils(fp2int_write_item)

  function new(string name = "fp2int_write_item");
    super.new(name);
  endfunction
endclass

class fp2int_resp_item extends bb_blink_resp_item;
  `uvm_object_utils(fp2int_resp_item)

  function new(string name = "fp2int_resp_item");
    super.new(name);
  endfunction
endclass
