class matadd_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(matadd_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, matadd_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, matadd_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, matadd_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, matadd_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, matadd_scoreboard) resp_imp;

  bb_blink_mem_model #(2, 1) mem_model;
  matadd_cmd_item stim;
  int unsigned read_count[2];
  int unsigned write_count;
  int unsigned cmd_count;
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
    matadd_cmd_item source;
    matadd_cmd_item clone;
    if (stim != null) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(source, item)) `uvm_fatal("SCB", "stimulus is not matadd_cmd_item")
    if (!$cast(clone, source.clone())) `uvm_fatal("SCB", "failed to clone stimulus")
    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    stim = clone;
    mem_model.clear_mem();
    for (int group = 0; group < stim.group_count; group++) begin
      for (int line = 0; line < stim.lines; line++) begin
        mem_model.load_word_g(stim.op1_bank, group, line, stim.a_word(group, line));
        mem_model.load_word_g(stim.op2_bank, group, line, stim.b_word(group, line));
      end
    end
    mem_model.arm();
  endfunction

  function matadd_cmd_item current_stim(string source);
    if (stim == null) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", source))
    return stim;
  endfunction

  function int unsigned total_lines();
    return current_stim("COUNT").lines * current_stim("COUNT").group_count;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    matadd_cmd_item expected;
    expected = current_stim("CMD");
    if (item.bid !== expected.bid) `uvm_fatal("CMD", "bid mismatch")
    if (item.funct7 !== expected.funct7) `uvm_fatal("CMD", "funct7 mismatch")
    if (item.iter !== expected.iter) `uvm_fatal("CMD", "iter mismatch")
    if (item.op1_bank !== expected.op1_bank || item.op2_bank !== expected.op2_bank ||
        item.wr_bank !== expected.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (item.op1_col !== expected.group_count || item.op2_col !== expected.group_count ||
        item.wr_col !== expected.group_count)
      `uvm_fatal("CMD", "group field mismatch")
    if (item.rob_id !== expected.rob_id) `uvm_fatal("CMD", "rob_id mismatch")
    if (item.op1_en !== 1'b1 || item.op2_en !== 1'b1 || item.wr_spad_en !== 1'b1 ||
        item.op1_from_spad !== 1'b1 || item.op2_from_spad !== 1'b1)
      `uvm_fatal("CMD", "enable field mismatch")
    if (item.meta_bank !== 0 || item.rs1 !== 0 || item.rs2 !== 0 || item.special !== 0 ||
        item.is_sub !== 0 || item.sub_rob_id !== 0)
      `uvm_fatal("CMD", "reserved field mismatch")
    if (cmd_count != 0) `uvm_fatal("CMD", "extra command observed")
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    matadd_cmd_item expected;
    int unsigned index;
    int unsigned group;
    int unsigned line;
    expected = current_stim("READ");
    if (item.port < 0 || item.port > 1) `uvm_fatal("READ", "invalid read port")
    index = read_count[item.port];
    if (index >= total_lines()) `uvm_fatal("READ", "extra read observed")
    group = index / expected.lines;
    line  = index % expected.lines;
    if (item.rob_id !== expected.rob_id) `uvm_fatal("READ", "rob_id mismatch")
    if (item.bank_id !== (item.port == 0 ? expected.op1_bank : expected.op2_bank))
      `uvm_fatal("READ", "bank mismatch")
    if (item.group_id !== group || item.addr !== line)
      `uvm_fatal("READ", $sformatf(
                 "address mismatch port=%0d group=%0d line=%0d", item.port, item.group_id, item.addr
                 ))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    matadd_cmd_item expected;
    int unsigned group;
    int unsigned line;
    expected = current_stim("WRITE");
    if (write_count >= total_lines()) `uvm_fatal("WRITE", "extra write observed")
    group = write_count / expected.lines;
    line  = write_count % expected.lines;
    if (item.port != 0 || item.bank_id !== expected.wr_bank || item.rob_id !== expected.rob_id)
      `uvm_fatal("WRITE", "port, bank, or rob_id mismatch")
    if (item.group_id !== group || item.addr !== line)
      `uvm_fatal("WRITE", $sformatf("address mismatch group=%0d line=%0d", item.group_id, item.addr
                 ))
    if (item.mask !== 16'hffff) `uvm_fatal("WRITE", "mask mismatch")
    if (item.data !== expected.sum_word(group, line))
      `uvm_fatal("WRITE", $sformatf("data mismatch group=%0d line=%0d", group, line))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    matadd_cmd_item expected;
    expected = current_stim("RESP");
    if (resp_count != 0) `uvm_fatal("RESP", "extra response observed")
    if (item.rob_id !== expected.rob_id || item.is_sub !== 0 || item.sub_rob_id !== 0)
      `uvm_fatal("RESP", "response mismatch")
    resp_count++;
  endfunction

  function bit done();
    return cmd_count == 1 && read_count[0] == total_lines() && read_count[1] == total_lines() &&
           write_count == total_lines() && resp_count == 1;
  endfunction

  function void reset_counters();
    stim = null;
    read_count[0] = 0;
    read_count[1] = 0;
    write_count = 0;
    cmd_count = 0;
    resp_count = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete cmds=%0d reads=%0d/%0d writes=%0d resp=%0d",
                 cmd_count,
                 read_count[0],
                 read_count[1],
                 write_count,
                 resp_count
                 ))
  endfunction
endclass
