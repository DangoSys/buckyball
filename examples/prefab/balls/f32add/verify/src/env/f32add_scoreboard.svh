class f32add_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(f32add_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, f32add_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, f32add_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, f32add_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, f32add_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, f32add_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  f32add_cmd_item stim_q[$];
  int unsigned cmd_count;
  int unsigned read_count[2];
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
    f32add_cmd_item got;
    f32add_cmd_item clone;

    if (stim_q.size() != 0) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(got, item)) `uvm_fatal("SCB", "stim item is not f32add_cmd_item")
    if (!$cast(clone, got.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    stim_q.push_back(clone);

    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    mem_model.clear_mem();
    for (int i = 0; i < clone.iter; i++) begin
      mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group + clone.group), i,
                            clone.a_words[i]);
      // With first set B is not an operand: leaving it unloaded makes any read fail.
      if (!clone.first)
        mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group), i, clone.b_words[i]);
    end
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    f32add_cmd_item   stim;
    stim = current_stim("CMD");
    // Compare as plain BLink commands; the comparer rejects differing types.
    want = bb_blink_cmd_item::type_id::create("want");
    want.copy(stim);
    verification_pkg::evidence::checks++;
    if (!item.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", item.sprint(), want.sprint()))
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    f32add_cmd_item stim;
    int unsigned expect_bank;
    int unsigned expect_group;

    stim = current_stim("READ");
    if (item.port != 0 && item.port != 1)
      `uvm_fatal("READ", $sformatf("F32AddBall has two read ports, got port %0d", item.port))
    if (item.port == 1 && stim.first) `uvm_fatal("READ", "B read although first is set")
    if (read_count[item.port] >= stim.iter)
      `uvm_fatal("READ", $sformatf("extra read on port %0d", item.port))
    if (read_count[item.port] != write_count)
      `uvm_fatal("READ", $sformatf(
                 "port %0d read %0d before write %0d", item.port, read_count[item.port], write_count
                 ))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.port == 0) begin
      expect_bank  = stim.op1_bank;
      expect_group = stim.op1_group + stim.group;
    end else begin
      expect_bank  = stim.op2_bank;
      expect_group = stim.op2_group;
    end
    if (item.bank_id !== expect_bank[BB_BLINK_BANK_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, expect_bank))
    if (item.group_id !== expect_group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group))
    if (item.addr !== read_count[item.port])
      `uvm_fatal("READ", $sformatf(
                 "addr mismatch: got %0d exp %0d", item.addr, read_count[item.port]))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    f32add_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("F32AddBall has one write port, got %0d", item.port))
    if (write_count >= stim.iter) `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    if (read_count[0] != write_count + 1 || (!stim.first && read_count[1] != write_count + 1))
      `uvm_fatal("WRITE", $sformatf("write %0d before its operands", write_count))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.group_id !== stim.wr_group)
      `uvm_fatal("WRITE", $sformatf("group mismatch: got %0d exp %0d", item.group_id, stim.wr_group
                 ))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.addr !== write_count)
      `uvm_fatal("WRITE", $sformatf("addr mismatch: got %0d exp %0d", item.addr, write_count))
    verification_pkg::evidence::checks++;
    if (item.data !== stim.c_words[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at row %0d: got 0x%032h exp 0x%032h",
                 item.addr,
                 item.data,
                 stim.c_words[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    f32add_cmd_item stim;
    stim = current_stim("RESP");
    if (write_count != stim.iter)
      `uvm_fatal("RESP", $sformatf("response after %0d of %0d writes", write_count, stim.iter))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function f32add_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    return stim_q[0];
  endfunction

  function bit done();
    if (stim_q.size() == 0) return 1'b0;
    return cmd_count == 1 && read_count[0] == stim_q[0].iter &&
        read_count[1] == (stim_q[0].first ? 0 : stim_q[0].iter) &&
        write_count == stim_q[0].iter && resp_count == 1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count     = 0;
    read_count[0] = 0;
    read_count[1] = 0;
    write_count   = 0;
    resp_count    = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: cmds=%0d reads=%0d,%0d writes=%0d resp=%0d",
                 cmd_count,
                 read_count[0],
                 read_count[1],
                 write_count,
                 resp_count
                 ))
    stim_q.delete();
  endfunction
endclass
