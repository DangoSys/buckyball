class int2fp_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(int2fp_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, int2fp_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, int2fp_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, int2fp_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, int2fp_scoreboard) resp_imp;

  bb_blink_mem_model #(1, 1) mem_model;

  int2fp_cmd_item stim_q[$];
  bit [127:0] expected_words[INT2FP_MAX_ITER];
  int unsigned expected_reads;
  int unsigned expected_writes;
  int unsigned cmd_count;
  int unsigned read_count;
  int unsigned write_count;
  int unsigned resp_count;
  int unsigned expect_group;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp  = new("stim_imp", this);
    cmd_imp   = new("cmd_imp", this);
    read_imp  = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp  = new("resp_imp", this);
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    int2fp_cmd_item fitem;
    int2fp_cmd_item clone;
    int i;
    int row;
    int group;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(fitem, item)) begin
      `uvm_fatal("SCB", "stim item is not int2fp_cmd_item")
    end
    if (!$cast(clone, fitem.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    if (clone.is_i8()) begin
      for (row = 0; row < clone.iter; row++) begin
        for (group = 0; group < INT2FP_NUM_GROUPS; group++) begin
          mem_model.load_word_g(int'(clone.op1_bank), group, row,
                                clone.input_words[row*INT2FP_NUM_GROUPS+group]);
        end
      end
    end else begin
      for (i = 0; i < clone.iter; i++) begin
        mem_model.load_word(int'(clone.op1_bank), i, clone.input_words[i]);
      end
    end
    mem_model.arm();

    build_expected(clone);
    expected_writes = clone.iter;
    expected_reads = clone.is_i8() ? (clone.iter * INT2FP_NUM_GROUPS) : clone.iter;
    expect_group = 0;
  endfunction

  function void build_expected(int2fp_cmd_item item);
    if (item.is_i8()) begin
      for (int row = 0; row < item.iter; row++) begin
        bit [127:0] packed_word = '0;
        for (int group = 0; group < INT2FP_NUM_GROUPS; group++) begin
          bit [127:0] src = item.input_words[row*INT2FP_NUM_GROUPS+group];
          for (int lane = 0; lane < 4; lane++) begin
            int signed v = $signed(src[lane*32+:32]);
            bit [7:0] q = int2fp_ref_i8(v, item.scale_bits);
            packed_word[group*32+lane*8+:8] = q;
          end
        end
        expected_words[row] = packed_word;
      end
    end else begin
      for (int w = 0; w < item.iter; w++) begin
        for (int e = 0; e < 4; e++) begin
          int signed v = $signed(item.input_words[w][e*32+:32]);
          expected_words[w][e*32+:32] = int2fp_ref_fp32(v, item.scale_bits);
        end
      end
    end
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int2fp_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, int2fp_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.iter !== exp.iter)
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d exp %0d", got.iter, exp.iter))
    if (got.special[31:0] !== exp.scale_bits)
      `uvm_fatal("CMD", $sformatf(
                 "scale mismatch: got 0x%08h exp 0x%08h", got.special[31:0], exp.scale_bits))
    if (got.special[33:32] !== exp.output_mode[1:0])
      `uvm_fatal("CMD", $sformatf(
                 "output_mode mismatch: got %0d exp %0d", got.special[33:32], exp.output_mode))
    if (got.op1_bank !== exp.op1_bank || got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
  endfunction

  function void write_read(bb_blink_read_item item);
    int2fp_cmd_item stim;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    if (stim.is_i8()) begin
      expect_addr = read_count / INT2FP_NUM_GROUPS;
      if (item.group_id !== expect_group[4:0])
        `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group
                   ))
      if (item.addr !== expect_addr[9:0])
        `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
      expect_group = (expect_group + 1) % INT2FP_NUM_GROUPS;
    end else begin
      if (item.group_id !== 5'd0)
        `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp 0", item.group_id))
      if (item.addr !== read_count[9:0])
        `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, read_count))
    end

    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    int2fp_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== '0)
      `uvm_fatal("WRITE", $sformatf("group_id mismatch: got %0d exp 0", item.group_id))
    if (item.addr !== write_count[9:0])
      `uvm_fatal("WRITE", $sformatf("addr mismatch: got %0d exp %0d", item.addr, write_count))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.data !== expected_words[item.addr])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at addr %0d: got 0x%032h exp 0x%032h",
                 item.addr,
                 item.data,
                 expected_words[item.addr]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    int2fp_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    if (item.sub_rob_id !== 8'h00)
      `uvm_fatal("RESP", $sformatf("sub_rob_id mismatch: got 0x%0h", item.sub_rob_id))
    resp_count++;
  endfunction

  function int2fp_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return cmd_count == 1 &&
           read_count == expected_reads &&
           write_count == expected_writes &&
           resp_count == 1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count = 0;
    read_count = 0;
    write_count = 0;
    resp_count = 0;
    expected_reads = 0;
    expected_writes = 0;
    expect_group = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d reads=%0d/%0d writes=%0d/%0d resp=%0d",
                                  cmd_count, read_count, expected_reads, write_count,
                                  expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
