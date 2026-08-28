class smatmul_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(smatmul_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, smatmul_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, smatmul_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, smatmul_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, smatmul_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, smatmul_scoreboard) resp_imp;

  bb_blink_mem_model #(2, 2) mem_model;

  smatmul_cmd_item stim_q[$];
  int unsigned exp_group[$];
  int unsigned exp_addr[$];
  bit [127:0] exp_data[$];
  bit [15:0] exp_mask[$];
  bit exp_seen[$];
  int unsigned expected_writes;
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
    smatmul_cmd_item mitem;
    smatmul_cmd_item clone;
    int i;
    longint unsigned d_lo;
    longint unsigned d_hi;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(mitem, item)) begin
      `uvm_fatal("SCB", "stim item is not smatmul_cmd_item")
    end
    if (!$cast(clone, mitem.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (i = 0; i < clone.num_a_words; i++) begin
      mem_model.load_word(int'(clone.op1_bank), i, clone.a_words[i]);
    end
    for (i = 0; i < clone.num_b_words; i++) begin
      mem_model.load_word(int'(clone.op2_bank), i, clone.b_words[i]);
    end
    mem_model.arm();

    exp_group.delete();
    exp_addr.delete();
    exp_data.delete();
    exp_mask.delete();
    exp_seen.delete();
    for (i = 0; i < clone.num_writes; i++) begin
      exp_group.push_back(smatmul_case_write_group(i));
      exp_addr.push_back(smatmul_case_write_addr(i));
      d_lo = smatmul_case_write_data_lo(i);
      d_hi = smatmul_case_write_data_hi(i);
      exp_data.push_back({d_hi, d_lo});
      exp_mask.push_back(smatmul_case_write_mask(i) [15:0]);
      exp_seen.push_back(1'b0);
    end
    expected_writes = clone.num_writes;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    smatmul_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, smatmul_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.op1_bank !== exp.op1_bank || got.op2_bank !== exp.op2_bank ||
        got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.op2_col !== exp.op2_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
    if (got.rs1 !== exp.rs1)
      `uvm_fatal("CMD", $sformatf("rs1 mismatch: got 0x%016h exp 0x%016h", got.rs1, exp.rs1))
    if (got.rs2 !== exp.rs2)
      `uvm_fatal("CMD", $sformatf("rs2 mismatch: got 0x%016h exp 0x%016h", got.rs2, exp.rs2))
    if (got.iter !== 34'd0) `uvm_fatal("CMD", "iter must be 0")
    if (got.op1_en !== 1'b1 || got.op2_en !== 1'b1 || got.wr_spad_en !== 1'b1)
      `uvm_fatal("CMD", "enable field mismatch")
    if (got.op1_from_spad !== 1'b1 || got.op2_from_spad !== 1'b1)
      `uvm_fatal("CMD", "from_spad field mismatch")
  endfunction

  function void write_read(bb_blink_read_item item);
    smatmul_cmd_item stim;
    stim = current_stim("READ");
    if (item.bank_id == stim.op1_bank) begin
      if (item.addr >= stim.num_a_words[9:0])
        `uvm_fatal("READ", $sformatf(
                   "op1 addr out of range: got %0d max %0d", item.addr, stim.num_a_words))
    end else if (item.bank_id == stim.op2_bank) begin
      if (item.addr >= stim.num_b_words[9:0])
        `uvm_fatal("READ", $sformatf(
                   "op2 addr out of range: got %0d max %0d", item.addr, stim.num_b_words))
    end else begin
      `uvm_fatal(
          "READ", $sformatf(
          "bank mismatch: got %0d op1=%0d op2=%0d", item.bank_id, stim.op1_bank, stim.op2_bank))
    end
    if (item.port < 0 || item.port > 1)
      `uvm_fatal("READ", $sformatf("unexpected read port %0d", item.port))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    smatmul_cmd_item stim;
    int i;
    bit found;
    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (write_count >= expected_writes) `uvm_fatal("WRITE", "extra write observed")

    found = 1'b0;
    for (i = 0; i < expected_writes; i++) begin
      if (exp_seen[i]) continue;
      if (item.group_id === exp_group[i][4:0] &&
          item.addr === exp_addr[i][9:0] &&
          item.mask === exp_mask[i] &&
          item.data === exp_data[i]) begin
        exp_seen[i] = 1'b1;
        found = 1'b1;
        break;
      end
    end
    if (!found) begin
      `uvm_fatal(
          "WRITE",
          $sformatf(
              "no matching expected write for port=%0d group=%0d addr=%0d mask=0x%04h data=0x%032h",
              item.port, item.group_id, item.addr, item.mask, item.data))
    end
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    smatmul_cmd_item stim;
    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    resp_count++;
  endfunction

  function smatmul_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return cmd_count == 1 &&
           read_count[0] > 0 &&
           read_count[1] > 0 &&
           write_count == expected_writes &&
           resp_count == 1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    exp_group.delete();
    exp_addr.delete();
    exp_data.delete();
    exp_mask.delete();
    exp_seen.delete();
    cmd_count = 0;
    read_count[0] = 0;
    read_count[1] = 0;
    write_count = 0;
    resp_count = 0;
    expected_writes = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d r0=%0d r1=%0d writes=%0d/%0d resp=%0d",
                                  cmd_count, read_count[0], read_count[1], write_count,
                                  expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
