class im2col_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(im2col_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, im2col_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, im2col_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, im2col_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, im2col_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, im2col_scoreboard) resp_imp;

  bb_blink_mem_model #(1, 1) mem_model;

  im2col_cmd_item stim_q[$];
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
    im2col_cmd_item titem;
    im2col_cmd_item clone;

    if (stim_q.size() != 0) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(titem, item)) `uvm_fatal("SCB", "stim item is not im2col_cmd_item")
    if (!$cast(clone, titem.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    stim_q.push_back(clone);

    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    mem_model.clear_mem();
    foreach (clone.src_data[i])
    mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group) + clone.src_group[i],
                          clone.src_addr[i], clone.src_data[i]);
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    im2col_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, im2col_cmd_item exp);
    // Compare as plain BLink commands; the comparer rejects differing types.
    bb_blink_cmd_item want = bb_blink_cmd_item::type_id::create("want");
    want.copy(exp);
    verification_pkg::evidence::checks++;
    if (!got.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", got.sprint(), want.sprint()))
  endfunction

  function void write_read(bb_blink_read_item item);
    im2col_cmd_item stim;
    int unsigned group;
    stim = current_stim("READ");
    if (read_count >= stim.read_addr.size())
      `uvm_fatal("READ", $sformatf("extra read %0d", read_count))
    group = stim.op1_group + stim.read_group[read_count];
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf(
                 "read %0d group mismatch: got %0d exp %0d", read_count, item.group_id, group))
    if (item.addr !== stim.read_addr[read_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("READ", $sformatf(
                 "read %0d addr mismatch: got %0d exp %0d",
                 read_count,
                 item.addr,
                 stim.read_addr[read_count]
                 ))
    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    im2col_cmd_item stim;
    int unsigned group;
    stim = current_stim("WRITE");
    if (write_count >= stim.dst_addr.size())
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    group = stim.wr_group + stim.dst_group[write_count];
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("WRITE", $sformatf(
                 "write %0d group mismatch: got %0d exp %0d", write_count, item.group_id, group))
    if (item.addr !== stim.dst_addr[write_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("WRITE", $sformatf(
                 "write %0d addr mismatch: got %0d exp %0d",
                 write_count,
                 item.addr,
                 stim.dst_addr[write_count]
                 ))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.data !== stim.dst_data[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "write %0d data mismatch: got 0x%032h exp 0x%032h",
                 write_count,
                 item.data,
                 stim.dst_data[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    im2col_cmd_item stim;
    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function im2col_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return stim_q.size() == 1 && cmd_count == 1 && read_count == stim_q[0].read_addr.size() &&
        write_count == stim_q[0].dst_addr.size() && resp_count == 1;
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
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d reads=%0d writes=%0d resp=%0d", cmd_count,
                                  read_count, write_count, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
