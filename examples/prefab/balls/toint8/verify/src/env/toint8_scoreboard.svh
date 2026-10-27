class toint8_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(toint8_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, toint8_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, toint8_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, toint8_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, toint8_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, toint8_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  toint8_cmd_item stim_q[$];
  int unsigned expected_reads;
  int unsigned expected_writes;
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
    toint8_cmd_item got;
    toint8_cmd_item clone;
    int i;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(got, item)) begin
      `uvm_fatal("SCB", "stim item is not toint8_cmd_item")
    end
    if (!$cast(clone, got.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    // Group g of every operand is channel block g with the same rows.
    for (i = 0; i < clone.num_src_words; i++) begin
      mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group) + i / int'(clone.iter),
                            int'(clone.input_base) + i % int'(clone.iter), clone.src_words[i]);
    end
    for (i = 0; i < clone.num_scale_words; i++) begin
      mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group) + i / 4, i % 4,
                            clone.scale_words[i]);
    end
    mem_model.arm();

    expected_writes = clone.num_dst_words;
    expected_reads  = clone.num_scale_words + clone.num_src_words;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    toint8_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, toint8_cmd_item exp);
    // Compare as plain BLink commands; the comparer rejects differing types.
    bb_blink_cmd_item want = bb_blink_cmd_item::type_id::create("want");
    want.copy(exp);
    verification_pkg::evidence::checks++;
    if (!got.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", got.sprint(), want.sprint()))
  endfunction

  function void write_read(bb_blink_read_item item);
    toint8_cmd_item stim;
    int unsigned scale_reads;
    int unsigned group_reads;
    int unsigned group;
    int unsigned index;
    int unsigned expect_bank;
    int unsigned expect_group;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.port != 0)
      `uvm_fatal("READ", $sformatf("ToInt8Ball has one read port, got port %0d", item.port))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    // Each group reads its scale rows (I32 only), then its input rows.
    scale_reads = stim.num_scale_words / stim.op1_col;
    group_reads = scale_reads + stim.iter;
    group       = read_count / group_reads;
    index       = read_count % group_reads;
    if (index < scale_reads) begin
      expect_bank  = stim.op2_bank;
      expect_group = stim.op2_group + group;
      expect_addr  = index;
    end else begin
      expect_bank  = stim.op1_bank;
      expect_group = stim.op1_group + group;
      expect_addr  = stim.input_base + (index - scale_reads);
    end
    if (item.bank_id !== expect_bank[BB_BLINK_BANK_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, expect_bank))
    if (item.group_id !== expect_group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group))
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    toint8_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("ToInt8Ball has one write port, got port %0d", item.port))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== stim.wr_group + write_count / (stim.iter >> 2))
      `uvm_fatal("WRITE", $sformatf(
                 "group mismatch: got %0d exp %0d",
                 item.group_id,
                 stim.wr_group + write_count / (stim.iter >> 2)
                 ))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (write_count >= stim.num_dst_words)
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    if (item.addr !== stim.dst_addr[write_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("WRITE", $sformatf(
                 "addr mismatch: got %0d exp %0d", item.addr, stim.dst_addr[write_count]))
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
    toint8_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function toint8_cmd_item current_stim(string tag);
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
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d reads=%0d/%0d writes=%0d/%0d resp=%0d",
                                  cmd_count, read_count, expected_reads, write_count,
                                  expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
