class transpose_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(transpose_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, transpose_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, transpose_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, transpose_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, transpose_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, transpose_scoreboard) resp_imp;

  bb_blink_mem_model #(1, 1) mem_model;

  transpose_cmd_item stim_q[$];
  bit [127:0] expected_dst[TRANSPOSE_MAX_WORDS];
  int unsigned expected_writes;
  int unsigned expected_reads;
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
    transpose_cmd_item titem;
    transpose_cmd_item clone;
    int i;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(titem, item)) begin
      `uvm_fatal("SCB", "stim item is not transpose_cmd_item")
    end
    if (!$cast(clone, titem.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (i = 0; i < clone.num_src_words; i++) begin
      mem_model.load_word(int'(clone.op1_bank), i, clone.src_words[i]);
    end
    mem_model.arm();

    for (i = 0; i < clone.num_dst_words; i++) begin
      expected_dst[i] = clone.dst_words[i];
    end
    expected_writes = clone.num_dst_words;
    expected_reads  = clone.num_src_words;
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    transpose_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, transpose_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.iter !== exp.iter)
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d exp %0d", got.iter, exp.iter))
    if (got.op1_bank !== exp.op1_bank || got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
    if (got.rs2[7:0] !== exp.elem_bits[7:0])
      `uvm_fatal("CMD", $sformatf(
                 "elem_bits mismatch: got %0d exp %0d", got.rs2[7:0], exp.elem_bits[7:0]))
  endfunction

  function void write_read(bb_blink_read_item item);
    transpose_cmd_item stim;
    stim = current_stim("READ");
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== '0)
      `uvm_fatal("READ", $sformatf("group_id mismatch: got %0d exp 0", item.group_id))
    // DUT may re-read the same bank word across output columns (cache refill).
    if (item.addr >= stim.num_src_words[6:0])
      `uvm_fatal("READ", $sformatf(
                 "addr out of range: got %0d max %0d", item.addr, stim.num_src_words))
    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    transpose_cmd_item stim;
    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== '0)
      `uvm_fatal("WRITE", $sformatf("group_id mismatch: got %0d exp 0", item.group_id))
    if (item.addr !== write_count[6:0])
      `uvm_fatal("WRITE", $sformatf("addr mismatch: got %0d exp %0d", item.addr, write_count))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.data !== expected_dst[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at %0d: got 0x%032h exp 0x%032h",
                 write_count,
                 item.data,
                 expected_dst[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    transpose_cmd_item stim;
    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    resp_count++;
  endfunction

  function transpose_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return cmd_count == 1 && read_count > 0 && write_count == expected_writes && resp_count == 1;
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
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d reads=%0d writes=%0d/%0d resp=%0d",
                                  cmd_count, read_count, write_count, expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
