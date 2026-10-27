class mxquant_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(mxquant_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxquant_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxquant_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, mxquant_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, mxquant_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, mxquant_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  mxquant_cmd_item stim_q[$];
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
    mxquant_cmd_item got;
    mxquant_cmd_item clone;

    if (stim_q.size() != 0) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(got, item)) `uvm_fatal("SCB", "stim item is not mxquant_cmd_item")
    if (!$cast(clone, got.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    stim_q.push_back(clone);
    if (mem_model == null) `uvm_fatal("SCB", "mem_model handle not set")
    mem_model.clear_mem();
    foreach (clone.preload_data[i])
    mem_model.load_word_g(clone.preload_bank[i], clone.preload_group[i], clone.preload_addr[i],
                          clone.preload_data[i]);
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    mxquant_cmd_item  stim;
    bb_blink_cmd_item want;
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
    mxquant_cmd_item stim;

    stim = current_stim("READ");
    if (item.port != 0)
      `uvm_fatal("READ", $sformatf("MxquantBall has one read port, got port %0d", item.port))
    if (read_count >= stim.read_addr.size())
      `uvm_fatal("READ", $sformatf("extra read %0d", read_count))
    verification_pkg::evidence::checks++;
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.group_id !== stim.op1_group)
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, stim.op1_group
                 ))
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
    mxquant_cmd_item stim;
    bit [127:0] bytes;

    stim = current_stim("WRITE");
    if (item.port != 0)
      `uvm_fatal("WRITE", $sformatf("MxquantBall has one write port, got port %0d", item.port))
    if (write_count >= stim.write_addr.size())
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    verification_pkg::evidence::checks++;
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.group_id !== stim.wr_group)
      `uvm_fatal("WRITE", $sformatf("group mismatch: got %0d exp %0d", item.group_id, stim.wr_group
                 ))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.addr !== stim.write_addr[write_count][BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("WRITE", $sformatf(
                 "write %0d addr mismatch: got %0d exp %0d",
                 write_count,
                 item.addr,
                 stim.write_addr[write_count]
                 ))
    if (item.mask !== stim.write_mask[write_count])
      `uvm_fatal("WRITE", $sformatf(
                 "write %0d mask mismatch: got 0x%04h exp 0x%04h",
                 write_count,
                 item.mask,
                 stim.write_mask[write_count]
                 ))
    for (int b = 0; b < 16; b++) bytes[b*8+:8] = {8{item.mask[b]}};
    if ((item.data & bytes) !== (stim.write_data[write_count] & bytes))
      `uvm_fatal("SCB", $sformatf(
                 "write %0d data mismatch at addr %0d: got 0x%032h exp 0x%032h",
                 write_count,
                 item.addr,
                 item.data,
                 stim.write_data[write_count]
                 ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    mxquant_cmd_item stim;
    stim = current_stim("RESP");
    verification_pkg::evidence::checks++;
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (!complete_io())
      `uvm_fatal("RESP", $sformatf(
                 "response before all traffic: reads=%0d writes=%0d", read_count, write_count))
    resp_count++;
  endfunction

  function mxquant_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    return stim_q[0];
  endfunction

  function bit complete_io();
    mxquant_cmd_item stim = stim_q[0];
    return read_count == stim.read_addr.size() && write_count == stim.write_addr.size();
  endfunction

  function bit done();
    return stim_q.size() == 1 && cmd_count == 1 && resp_count == 1 && complete_io();
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
                 "incomplete: stim=%0d cmds=%0d reads=%0d writes=%0d resp=%0d",
                 stim_q.size(),
                 cmd_count,
                 read_count,
                 write_count,
                 resp_count
                 ))
    stim_q.delete();
  endfunction
endclass
