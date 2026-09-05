class int8mul_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(int8mul_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, int8mul_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int8mul_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, int8mul_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, int8mul_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, int8mul_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  int8mul_cmd_item stim_q[$];
  int unsigned expected_reads[`BB_IN_BW];
  int unsigned expected_writes;
  int unsigned cmd_count;
  int unsigned read_count[`BB_IN_BW];
  int unsigned write_count;
  int unsigned resp_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp  = new("stim_imp", this);
    cmd_imp   = new("cmd_imp", this);
    read_imp  = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp  = new("resp_imp", this);
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    int8mul_cmd_item got;
    int8mul_cmd_item clone;
    int i;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(got, item)) begin
      `uvm_fatal("SCB", "stim item is not int8mul_cmd_item")
    end
    if (!$cast(clone, got.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (i = 0; i < clone.num_gate_words; i++) begin
      mem_model.load_word(int'(clone.op1_bank), i, clone.gate_words[i]);
    end
    for (i = 0; i < clone.num_input_words; i++) begin
      mem_model.load_word(int'(clone.op2_bank), i, clone.input_words[i]);
    end
    mem_model.arm();

    expected_writes   = clone.num_dst_words;
    expected_reads[0] = 1;
    expected_reads[1] = clone.iter;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int8mul_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, int8mul_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.iter !== exp.iter)
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d exp %0d", got.iter, exp.iter))
    if (got.op1_bank !== exp.op1_bank || got.op2_bank !== exp.op2_bank ||
        got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.op2_col !== exp.op2_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rs1 !== exp.rs1 || got.rs2 !== exp.rs2) `uvm_fatal("CMD", "rs field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
    if (got.rs2[63:38] != 0) `uvm_fatal("CMD", $sformatf("reserved rs2 bits set: 0x%016h", got.rs2))
  endfunction

  function void write_read(bb_blink_read_item item);
    int8mul_cmd_item stim;
    int unsigned expect_bank;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.port != 0 && item.port != 1)
      `uvm_fatal("READ", $sformatf("Int8MulBall has two read ports, got port %0d", item.port))
    if (item.group_id !== 5'd0)
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp 0", item.group_id))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    if (item.port == 0) begin
      expect_bank = stim.op1_bank;
      expect_addr = stim.gate_row;
      if (read_count[0] != 0) `uvm_fatal("READ", "Int8MulBall gate bank read more than once")
    end else begin
      expect_bank = stim.op2_bank;
      expect_addr = read_count[1];
    end
    if (item.bank_id !== expect_bank[4:0])
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, expect_bank))
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    int8mul_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("Int8MulBall has one write port, got port %0d", item.port))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== 5'd0)
      `uvm_fatal("WRITE", $sformatf("group mismatch: got %0d exp 0", item.group_id))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (write_count >= stim.num_dst_words)
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    if (item.addr !== write_count[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("WRITE", $sformatf("addr mismatch: got %0d exp %0d", item.addr, write_count))
    if (item.data !== stim.dst_words[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at addr %0d: got 0x%032h exp 0x%032h",
                 item.addr,
                 item.data,
                 stim.dst_words[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    int8mul_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    if (item.sub_rob_id !== 8'h00)
      `uvm_fatal("RESP", $sformatf("sub_rob_id mismatch: got 0x%0h", item.sub_rob_id))
    resp_count++;
  endfunction

  function int8mul_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    end
    return stim_q[0];
  endfunction

  function bit done();
    if (cmd_count != 1 || write_count != expected_writes || resp_count != 1) return 1'b0;
    for (int p = 0; p < `BB_IN_BW; p++) begin
      if (read_count[p] != expected_reads[p]) return 1'b0;
    end
    return 1'b1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count       = 0;
    write_count     = 0;
    resp_count      = 0;
    expected_writes = 0;
    for (int p = 0; p < `BB_IN_BW; p++) begin
      read_count[p]     = 0;
      expected_reads[p] = 0;
    end
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf(
                            "incomplete: cmds=%0d reads=%0d/%0d,%0d/%0d writes=%0d/%0d resp=%0d",
                            cmd_count, read_count[0], expected_reads[0], read_count[1],
                            expected_reads[1], write_count, expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
