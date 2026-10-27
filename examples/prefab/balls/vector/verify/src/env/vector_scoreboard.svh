class vector_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(vector_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, vector_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, vector_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, vector_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, vector_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, vector_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  vector_cmd_item stim_q[$];
  int unsigned cmd_count;
  int unsigned read_count[2];
  int unsigned write_count[VECMAT16_GROUPS];
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
    vector_cmd_item got;
    vector_cmd_item clone;

    if (stim_q.size() != 0) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(got, item)) `uvm_fatal("SCB", "stim item is not vector_cmd_item")
    if (!$cast(clone, got.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    stim_q.push_back(clone);

    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    mem_model.clear_mem();
    for (int g = 0; g < VECMAT16_GROUPS; g++)
    for (int row = 0; row < VECMAT16_ROWS; row++)
    mem_model.load_word_g(int'(clone.wr_read_bank), int'(clone.wr_group) + g, row,
                          clone.c_old_words[g*VECMAT16_ROWS+row]);
    for (int k = 0; k < clone.iter; k++) begin
      mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group), k, clone.at_words[k]);
      mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group), k, clone.b_words[k]);
    end
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    vector_cmd_item   stim;
    stim = current_stim("CMD");
    // Compare as plain BLink commands; the comparer rejects differing types.
    want = bb_blink_cmd_item::type_id::create("want");
    want.copy(stim);
    verification_pkg::evidence::checks++;
    if (!item.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", item.sprint(), want.sprint()))
    cmd_count++;
  endfunction

  // Port p first reads the old C, row by row, group p then group 2 + p past
  // the wr base; then port 0 streams A^T and port 1 streams B, one row per
  // reduction step, at their own group bases.
  function void write_read(bb_blink_read_item item);
    vector_cmd_item stim;
    int unsigned n;
    int unsigned expect_bank;
    int unsigned expect_group;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.port != 0 && item.port != 1)
      `uvm_fatal("READ", $sformatf("VecBall reads on two ports, got port %0d", item.port))
    n = read_count[item.port];
    if (n >= 2 * VECMAT16_ROWS + stim.iter)
      `uvm_fatal("READ", $sformatf("extra read on port %0d", item.port))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (n < 2 * VECMAT16_ROWS) begin
      expect_bank  = stim.wr_read_bank;
      expect_group = stim.wr_group + 2 * (n % 2) + item.port;
      expect_addr  = n / 2;
    end else begin
      expect_bank  = item.port == 0 ? stim.op1_bank : stim.op2_bank;
      expect_group = item.port == 0 ? stim.op1_group : stim.op2_group;
      expect_addr  = n - 2 * VECMAT16_ROWS;
    end
    if (item.bank_id !== expect_bank[BB_BLINK_BANK_ID_W-1:0])
      `uvm_fatal("READ", $sformatf(
                 "port %0d read %0d bank mismatch: got %0d exp %0d",
                 item.port,
                 n,
                 item.bank_id,
                 expect_bank
                 ))
    if (item.group_id !== expect_group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf(
                 "port %0d read %0d group mismatch: got %0d exp %0d",
                 item.port,
                 n,
                 item.group_id,
                 expect_group
                 ))
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal(
          "READ", $sformatf(
          "port %0d read %0d addr mismatch: got %0d exp %0d", item.port, n, item.addr, expect_addr))
    read_count[item.port]++;
  endfunction

  // Write port g stores group g of C past the wr base, rows in order, after
  // every operand read.
  function void write_write(bb_blink_write_item item);
    vector_cmd_item stim;
    int unsigned g;
    int unsigned row;

    stim = current_stim("WRITE");
    if (item.port < 0 || item.port >= VECMAT16_GROUPS)
      `uvm_fatal("WRITE", $sformatf("VecBall writes on four ports, got port %0d", item.port))
    g   = item.port;
    row = write_count[g];
    if (row >= VECMAT16_ROWS) `uvm_fatal("WRITE", $sformatf("extra write on port %0d", g))
    for (int p = 0; p < 2; p++)
    if (read_count[p] != 2 * VECMAT16_ROWS + stim.iter)
      `uvm_fatal("WRITE", $sformatf("write before port %0d finished reading", p))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.group_id !== stim.wr_group + g)
      `uvm_fatal("WRITE", $sformatf(
                 "group mismatch: got %0d exp %0d", item.group_id, stim.wr_group + g))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.addr !== row)
      `uvm_fatal("WRITE", $sformatf("port %0d addr mismatch: got %0d exp %0d", g, item.addr, row))
    verification_pkg::evidence::checks++;
    if (item.data !== stim.c_new_words[g*VECMAT16_ROWS+row])
      `uvm_fatal("SCB", $sformatf(
                 "C mismatch at row %0d group %0d: got 0x%032h exp 0x%032h",
                 row,
                 g,
                 item.data,
                 stim.c_new_words[g*VECMAT16_ROWS+row]
                 ))
    write_count[g]++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    vector_cmd_item stim;
    stim = current_stim("RESP");
    for (int g = 0; g < VECMAT16_GROUPS; g++)
    if (write_count[g] != VECMAT16_ROWS)
      `uvm_fatal("RESP", $sformatf("response after %0d writes on port %0d", write_count[g], g))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function vector_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    return stim_q[0];
  endfunction

  function bit done();
    if (stim_q.size() == 0 || cmd_count != 1 || resp_count != 1) return 1'b0;
    for (int p = 0; p < 2; p++)
    if (read_count[p] != 2 * VECMAT16_ROWS + stim_q[0].iter) return 1'b0;
    for (int g = 0; g < VECMAT16_GROUPS; g++) if (write_count[g] != VECMAT16_ROWS) return 1'b0;
    return 1'b1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count  = 0;
    resp_count = 0;
    for (int p = 0; p < 2; p++) read_count[p] = 0;
    for (int g = 0; g < VECMAT16_GROUPS; g++) write_count[g] = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: cmds=%0d reads=%0d,%0d writes=%0d,%0d,%0d,%0d resp=%0d",
                 cmd_count,
                 read_count[0],
                 read_count[1],
                 write_count[0],
                 write_count[1],
                 write_count[2],
                 write_count[3],
                 resp_count
                 ))
    stim_q.delete();
  endfunction
endclass
