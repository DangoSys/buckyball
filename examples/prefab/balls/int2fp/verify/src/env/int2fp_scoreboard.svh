class int2fp_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(int2fp_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, int2fp_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, int2fp_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, int2fp_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, int2fp_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  int2fp_cmd_item stim_q[$];
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
    int2fp_cmd_item got;
    int2fp_cmd_item clone;
    int i;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(got, item)) begin
      `uvm_fatal("SCB", "stim item is not int2fp_cmd_item")
    end
    if (!$cast(clone, got.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (i = 0; i < clone.num_src_words; i++) begin
      mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group + i / clone.iter),
                            i % clone.iter, clone.src_words[i]);
    end
    for (i = 0; i < clone.num_scale_words; i++) begin
      mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group + i / 4), i % 4,
                            clone.scale_words[i]);
    end
    mem_model.arm();

    expected_writes   = clone.num_dst_words;
    expected_reads[0] = clone.iter * clone.op1_col;
    expected_reads[1] = clone.iter * clone.op1_col;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int2fp_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, int2fp_cmd_item exp);
    // Compare as plain BLink commands; the comparer rejects differing types.
    bb_blink_cmd_item want = bb_blink_cmd_item::type_id::create("want");
    want.copy(exp);
    verification_pkg::evidence::checks++;
    if (!got.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", got.sprint(), want.sprint()))
  endfunction

  function void write_read(bb_blink_read_item item);
    int2fp_cmd_item stim;
    int unsigned expect_bank;
    int unsigned expect_group;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.port != 0 && item.port != 1)
      `uvm_fatal("READ", $sformatf("Int2FpBall has two read ports, got port %0d", item.port))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    if (item.port == 0) begin
      expect_bank  = stim.op1_bank;
      expect_group = stim.op1_group + read_count[0] / stim.iter;
      expect_addr  = read_count[0] % stim.iter;
    end else begin
      expect_bank  = stim.op2_bank;
      expect_group = stim.op2_group + read_count[1] / stim.iter;
      expect_addr  = (read_count[1] % stim.iter) & 32'h3;
    end
    if (item.bank_id !== expect_bank[BB_BLINK_BANK_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, expect_bank))
    if (item.group_id !== expect_group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group))
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    int2fp_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("Int2FpBall has one write port, got port %0d", item.port))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== stim.wr_group + write_count / stim.iter)
      `uvm_fatal(
          "WRITE", $sformatf(
          "group mismatch: got %0d exp %0d", item.group_id, stim.wr_group + write_count / stim.iter
          ))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (write_count >= stim.num_dst_words)
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    if (item.addr !== write_count % stim.iter)
      `uvm_fatal("WRITE", $sformatf(
                 "addr mismatch: got %0d exp %0d", item.addr, write_count % stim.iter))
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
    int2fp_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
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
           read_count[0] == expected_reads[0] &&
           read_count[1] == expected_reads[1] &&
           write_count == expected_writes &&
           resp_count == 1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count = 0;
    read_count[0] = 0;
    read_count[1] = 0;
    write_count = 0;
    resp_count = 0;
    expected_reads[0] = 0;
    expected_reads[1] = 0;
    expected_writes = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
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
