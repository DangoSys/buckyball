class smatmul_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(smatmul_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, smatmul_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, smatmul_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, smatmul_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, smatmul_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, smatmul_scoreboard) resp_imp;

  bb_blink_mem_model #(2, 1) mem_model;
  smatmul_cmd_item stim_q[$];
  int unsigned exp_addr[$];
  bit [127:0] exp_data[$];
  bit exp_seen[$];
  int unsigned expected_commands;
  int unsigned expected_reads[2];
  int unsigned expected_writes;
  int unsigned stim_count;
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
    smatmul_cmd_item got;
    smatmul_cmd_item clone;
    matrix_cmd_dpi_t data_cmd;
    longint unsigned lo;
    longint unsigned hi;
    if (!$cast(got, item) || !$cast(clone, got.clone()))
      `uvm_fatal("SCB", "failed to clone smatmul stimulus")
    if (got.command_index != stim_count) `uvm_fatal("SCB", "commands arrived out of order")

    if (stim_count == 0) begin
      if (got.kind != 0) `uvm_fatal("SCB", "first command must preload bias")
      if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
      expected_commands = got.command_count;
      mem_model.clear_mem();
      for (int i = 0; i < smatmul_case_bias_words(); i++) begin
        lo = smatmul_case_bias_word_lo(i);
        hi = smatmul_case_bias_word_hi(i);
        mem_model.load_word(0, i, {hi, lo});
      end
      for (int block = 0; block < expected_commands - 1; block++) begin
        smatmul_case_cmd(block + 1, data_cmd);
        for (int i = 0; i < smatmul_case_a_words(block); i++) begin
          lo = smatmul_case_a_word_lo(block, i);
          hi = smatmul_case_a_word_hi(block, i);
          mem_model.load_word(data_cmd.rs1_lo[9:0], i, {hi, lo});
        end
        for (int i = 0; i < smatmul_case_b_words(block); i++) begin
          lo = smatmul_case_b_word_lo(block, i);
          hi = smatmul_case_b_word_hi(block, i);
          mem_model.load_word(data_cmd.rs1_lo[19:10], i, {hi, lo});
        end
      end
      mem_model.arm();

      expected_writes = smatmul_case_num_writes();
      for (int i = 0; i < expected_writes; i++) begin
        exp_addr.push_back(smatmul_case_write_addr(i));
        lo = smatmul_case_write_data_lo(i);
        hi = smatmul_case_write_data_hi(i);
        exp_data.push_back({hi, lo});
        exp_seen.push_back(1'b0);
      end
    end else if (got.command_count != expected_commands) begin
      `uvm_fatal("SCB", "command count changed within a case")
    end

    expected_reads[0] += got.op1_words;
    expected_reads[1] += got.kind == 0 ? 0 : got.op2_words;
    stim_q.push_back(clone);
    stim_count++;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    smatmul_cmd_item exp;
    exp = current_stim("CMD");
    if (item.bid !== exp.bid || item.funct7 !== exp.funct7 ||
        item.op1_bank !== exp.op1_bank || item.op2_bank !== exp.op2_bank ||
        item.wr_bank !== exp.wr_bank || item.rs1 !== exp.rs1 ||
        item.rs2 !== exp.rs2 || item.iter !== exp.iter ||
        item.op1_col !== exp.op1_col || item.op2_col !== exp.op2_col ||
        item.wr_col !== exp.wr_col || item.op1_en !== exp.op1_en ||
        item.op2_en !== exp.op2_en || item.wr_spad_en !== exp.wr_spad_en)
      `uvm_fatal("CMD", "command differs from generated contract")
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    smatmul_cmd_item stim;
    stim = current_stim("READ");
    if (stim.kind == 0) begin
      if (item.port != 0 || item.bank_id != stim.op1_bank || item.addr >= 4)
        `uvm_fatal("READ", "invalid bias read")
    end else if (item.port == 0) begin
      if (item.bank_id != stim.op1_bank || item.addr >= stim.op1_words)
        `uvm_fatal("READ", "A read outside command footprint")
    end else if (item.port == 1) begin
      if (item.bank_id != stim.op2_bank || item.addr >= stim.op2_words)
        `uvm_fatal("READ", "B read outside command footprint")
    end else begin
      `uvm_fatal("READ", "unexpected read port")
    end
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    smatmul_cmd_item stim;
    int match_index = -1;
    stim = current_stim("WRITE");
    if (stim.kind != 1 || stim.rs2[25] != 1)
      `uvm_fatal("WRITE", "write before the last accumulation block")
    if (item.port != 0 || item.group_id != 0 || item.bank_id != stim.wr_bank ||
        item.mask != 16'hffff || item.rob_id != stim.rob_id)
      `uvm_fatal("WRITE", "write metadata violates SMatMul contract")
    for (int i = 0; i < expected_writes; i++) begin
      if (!exp_seen[i] && item.addr === exp_addr[i][9:0] && item.data === exp_data[i]) begin
        match_index = i;
        break;
      end
    end
    if (match_index < 0) `uvm_fatal("WRITE", "no matching expected result row")
    exp_seen[match_index] = 1'b1;
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    smatmul_cmd_item stim;
    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id || item.is_sub !== 1'b0)
      `uvm_fatal("RESP", "response metadata mismatch")
    void'(stim_q.pop_front());
    resp_count++;
  endfunction

  function smatmul_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed without a live command", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return expected_commands != 0 && stim_count == expected_commands &&
           cmd_count == expected_commands && resp_count == expected_commands &&
           stim_q.size() == 0 && read_count[0] == expected_reads[0] &&
           read_count[1] == expected_reads[1] &&
           write_count == expected_writes;
  endfunction

  function void reset_counters();
    stim_q.delete();
    exp_addr.delete();
    exp_data.delete();
    exp_seen.delete();
    expected_commands = 0;
    expected_reads[0] = 0;
    expected_reads[1] = 0;
    expected_writes = 0;
    stim_count = 0;
    cmd_count = 0;
    read_count[0] = 0;
    read_count[1] = 0;
    write_count = 0;
    resp_count = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: stim=%0d cmd=%0d resp=%0d reads=%0d/%0d,%0d/%0d writes=%0d/%0d",
                 stim_count,
                 cmd_count,
                 resp_count,
                 read_count[0],
                 expected_reads[0],
                 read_count[1],
                 expected_reads[1],
                 write_count,
                 expected_writes
                 ))
  endfunction
endclass
