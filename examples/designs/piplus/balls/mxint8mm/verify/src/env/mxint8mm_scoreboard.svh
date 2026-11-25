// Checks one command at a time: its reads stay on the A, B and old C banks,
// every C bank row the reference writes is written exactly once with the
// reference data, nothing else is written, and the response comes after all
// writes.
class mxint8mm_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(mxint8mm_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8mm_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8mm_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, mxint8mm_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, mxint8mm_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8mm_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  mxint8mm_cmd_item stim;
  bit [127:0] expected[int unsigned];
  bit written[int unsigned];
  int unsigned cmd_count;
  int unsigned read_count[2];
  int unsigned old_reads;
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

  // The first step of a case starts from an empty memory; later steps keep
  // what earlier steps wrote.
  function void write_stim(bb_blink_cmd_item item);
    mxint8mm_cmd_item got;
    if (stim != null) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(got, item)) `uvm_fatal("SCB", "stim item is not mxint8mm_cmd_item")
    if (!$cast(stim, got.clone())) `uvm_fatal("SCB", "failed to clone stimulus item")
    if (stim.step == 0) mem_model.clear_mem();
    foreach (stim.preload_data[i])
    mem_model.load_word_g(stim.preload_bank[i], stim.preload_group[i], stim.preload_addr[i],
                          stim.preload_data[i]);
    mem_model.arm();
    expected.delete();
    written.delete();
    foreach (stim.write_addr[i]) begin
      if (expected.exists(stim.write_addr[i]))
        `uvm_fatal("CASE", $sformatf("C row %0d written twice by the reference", stim.write_addr[i]
                   ))
      expected[stim.write_addr[i]] = stim.write_data[i];
    end
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    current("CMD");
    want = bb_blink_cmd_item::type_id::create("want");
    want.copy(stim);
    verification_pkg::evidence::checks++;
    if (!item.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", item.sprint(), want.sprint()))
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    bit a, b, old;
    current("READ");
    a = item.port == 0 && item.bank_id == stim.op1_bank && item.group_id == stim.op1_group;
    old = item.port == 0 && stim.read_old && item.bank_id == stim.wr_read_bank &&
        item.group_id == stim.wr_group;
    b = item.port == 1 && item.bank_id == stim.op2_bank && item.group_id == stim.op2_group;
    verification_pkg::evidence::checks++;
    if (!(a || b || old))
      `uvm_fatal("READ", $sformatf(
                 "port %0d read of bank %0d group %0d addr %0d outside A %0d/%0d, B %0d/%0d, old C %0d/%0d (read_old %0d)",
                 item.port,
                 item.bank_id,
                 item.group_id,
                 item.addr,
                 stim.op1_bank,
                 stim.op1_group,
                 stim.op2_bank,
                 stim.op2_group,
                 stim.wr_read_bank,
                 stim.wr_group,
                 stim.read_old
                 ))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (old && !a) old_reads++;
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    int unsigned addr;
    current("WRITE");
    addr = item.addr;
    verification_pkg::evidence::checks++;
    if (item.port != 0 || item.bank_id !== stim.wr_bank || item.group_id !== stim.wr_group ||
        item.rob_id !== stim.rob_id || item.mask !== 16'hffff)
      `uvm_fatal("WRITE", $sformatf(
                 "write port %0d bank %0d group %0d rob %0d mask 0x%04h, exp port 0 bank %0d group %0d rob %0d mask 0xffff",
                 item.port,
                 item.bank_id,
                 item.group_id,
                 item.rob_id,
                 item.mask,
                 stim.wr_bank,
                 stim.wr_group,
                 stim.rob_id
                 ))
    if (!expected.exists(addr))
      `uvm_fatal("WRITE", $sformatf("write to bank row %0d outside C", addr))
    if (written.exists(addr)) `uvm_fatal("WRITE", $sformatf("C bank row %0d written twice", addr))
    if (item.data !== expected[addr])
      `uvm_fatal("SCB", $sformatf(
                 "C bank row %0d mismatch (FP32 lanes 3..0): got %08h %08h %08h %08h exp %08h %08h %08h %08h",
                 addr,
                 item.data[127:96],
                 item.data[95:64],
                 item.data[63:32],
                 item.data[31:0],
                 expected[addr][127:96],
                 expected[addr][95:64],
                 expected[addr][63:32],
                 expected[addr][31:0]
                 ))
    written[addr] = 1'b1;
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    current("RESP");
    verification_pkg::evidence::checks++;
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (resp_count != 0) `uvm_fatal("RESP", "second response")
    if (write_count != expected.size())
      `uvm_fatal("RESP", $sformatf("response after %0d of %0d C rows", write_count, expected.size()
                 ))
    if (stim.read_old && old_reads == 0) `uvm_fatal("RESP", "old C or bias expected but none read")
    resp_count++;
  endfunction

  function void current(string tag);
    if (stim == null) `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
  endfunction

  function bit done();
    return stim != null && cmd_count == 1 && resp_count == 1;
  endfunction

  function void reset_counters();
    stim          = null;
    cmd_count     = 0;
    read_count[0] = 0;
    read_count[1] = 0;
    old_reads     = 0;
    write_count   = 0;
    resp_count    = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: cmds=%0d reads=%0d,%0d writes=%0d resp=%0d",
                 cmd_count,
                 read_count[0],
                 read_count[1],
                 write_count,
                 resp_count
                 ))
  endfunction
endclass
