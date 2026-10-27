class relu_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(relu_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, relu_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, relu_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, relu_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, relu_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, relu_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  relu_cmd_item stim_q[$];
  int unsigned cmd_count;
  int unsigned read_count;
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
    relu_cmd_item got;
    relu_cmd_item clone;

    if (stim_q.size() != 0) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(got, item)) `uvm_fatal("SCB", "stim item is not relu_cmd_item")
    if (!$cast(clone, got.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    stim_q.push_back(clone);

    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    mem_model.clear_mem();
    // Each row is read exactly once before it is rewritten.
    for (int i = 0; i < clone.num_words; i++)
    mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group + clone.op2_bank),
                          clone.rows[i], clone.src_words[i]);
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    relu_cmd_item stim;
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
    relu_cmd_item stim;
    int unsigned  group;

    stim  = current_stim("READ");
    group = stim.op1_group + stim.op2_bank;
    if (item.port != 0)
      `uvm_fatal("READ", $sformatf("ReluBall has one read port, got %0d", item.port))
    if (read_count >= stim.num_words) `uvm_fatal("READ", $sformatf("extra read %0d", read_count))
    if (read_count != write_count)
      `uvm_fatal("READ", $sformatf("read %0d issued before write %0d", read_count, write_count))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.group_id !== group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, group))
    if (item.addr !== stim.rows[read_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal(
          "READ", $sformatf(
          "addr mismatch at %0d: got %0d exp %0d", read_count, item.addr, stim.rows[read_count]))
    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    relu_cmd_item stim;
    int unsigned  group;

    stim  = current_stim("WRITE");
    group = stim.op1_group + stim.op2_bank;
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("ReluBall has one write port, got %0d", item.port))
    if (write_count >= read_count)
      `uvm_fatal("WRITE", $sformatf("write %0d before its read", write_count))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.group_id !== group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("WRITE", $sformatf("group mismatch: got %0d exp %0d", item.group_id, group))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.addr !== stim.rows[write_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal(
          "WRITE", $sformatf(
          "addr mismatch at %0d: got %0d exp %0d", write_count, item.addr, stim.rows[write_count]))
    verification_pkg::evidence::checks++;
    if (item.data !== stim.dst_words[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at row %0d: got 0x%032h exp 0x%032h",
                 item.addr,
                 item.data,
                 stim.dst_words[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    relu_cmd_item stim;
    stim = current_stim("RESP");
    if (write_count != stim.num_words)
      `uvm_fatal("RESP", $sformatf("response after %0d of %0d writes", write_count, stim.num_words))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function relu_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    return stim_q[0];
  endfunction

  function bit done();
    if (stim_q.size() == 0) return 1'b0;
    return cmd_count == 1 && read_count == stim_q[0].num_words &&
        write_count == stim_q[0].num_words && resp_count == 1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count   = 0;
    read_count  = 0;
    write_count = 0;
    resp_count  = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: cmds=%0d reads=%0d writes=%0d resp=%0d",
                 cmd_count,
                 read_count,
                 write_count,
                 resp_count
                 ))
    stim_q.delete();
  endfunction
endclass
