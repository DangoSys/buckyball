// Checks one command at a time against the Rust reference: every row of the
// input window is read exactly once, every bank row of the packed output is
// written exactly once with a full mask and the reference data, nothing else
// is touched, and the response comes after all of it.
class mxint8quant_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(mxint8quant_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8quant_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8quant_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, mxint8quant_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, mxint8quant_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8quant_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  mxint8quant_cmd_item stim;
  bit unread[int unsigned];
  bit [127:0] unwritten[int unsigned];
  int unsigned kind[int unsigned];
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
    if (stim != null) `uvm_fatal("SCB", "single outstanding command supported")
    if (!$cast(stim, item)) `uvm_fatal("SCB", "stim item is not mxint8quant_cmd_item")
    mem_model.clear_mem();
    foreach (stim.preload_addr[i]) begin
      mem_model.load_word_g(stim.op1_bank, stim.op1_group, stim.preload_addr[i],
                            stim.preload_data[i]);
      unread[stim.preload_addr[i]] = 1'b1;
    end
    foreach (stim.write_addr[i]) begin
      unwritten[stim.write_addr[i]] = stim.write_data[i];
      kind[stim.write_addr[i]] = stim.write_kind[i];
    end
    if (unread.size() != stim.preload_addr.size() || unwritten.size() != stim.write_addr.size())
      `uvm_fatal("SCB", "reference windows repeat a bank row")
    mem_model.arm();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    // Compare as plain BLink commands; the comparer rejects differing types.
    want = bb_blink_cmd_item::type_id::create("want");
    want.copy(stim);
    verification_pkg::evidence::checks++;
    if (!item.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", item.sprint(), want.sprint()))
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    verification_pkg::evidence::checks++;
    if (item.port != 0 || item.rob_id !== stim.rob_id || item.bank_id !== stim.op1_bank ||
        item.group_id !== stim.op1_group)
      `uvm_fatal("READ", $sformatf(
                 "read port %0d rob %0d bank %0d group %0d, expected 0/%0d/%0d/%0d",
                 item.port,
                 item.rob_id,
                 item.bank_id,
                 item.group_id,
                 stim.rob_id,
                 stim.op1_bank,
                 stim.op1_group
                 ))
    if (!unread.exists(item.addr))
      `uvm_fatal("READ", $sformatf("read of row %0d outside the input window or twice", item.addr))
    unread.delete(item.addr);
  endfunction

  function void write_write(bb_blink_write_item item);
    verification_pkg::evidence::checks++;
    if (item.port != 0 || item.rob_id !== stim.rob_id || item.bank_id !== stim.wr_bank ||
        item.group_id !== stim.wr_group)
      `uvm_fatal("WRITE", $sformatf(
                 "write port %0d rob %0d bank %0d group %0d, expected 0/%0d/%0d/%0d",
                 item.port,
                 item.rob_id,
                 item.bank_id,
                 item.group_id,
                 stim.rob_id,
                 stim.wr_bank,
                 stim.wr_group
                 ))
    if (!unwritten.exists(item.addr))
      `uvm_fatal("WRITE", $sformatf("write of row %0d outside the output window or twice", item.addr
                 ))
    if (item.mask !== 16'hffff)
      `uvm_fatal("WRITE", $sformatf(
                 "row %0d mask 0x%04h, expected a whole row", item.addr, item.mask))
    verification_pkg::evidence::comparisons++;
    if (item.data !== unwritten[item.addr])
      `uvm_fatal("SCB", $sformatf(
                 "%s row %0d: got 0x%032h exp 0x%032h",
                 kind[item.addr] == 0 ? "code" : "exponent",
                 item.addr,
                 item.data,
                 unwritten[item.addr]
                 ))
    unwritten.delete(item.addr);
  endfunction

  function void write_resp(bb_blink_resp_item item);
    verification_pkg::evidence::checks++;
    if (item.rob_id !== stim.rob_id || resp_count != 0)
      `uvm_fatal("RESP", $sformatf(
                 "response rob %0d, expected one for %0d", item.rob_id, stim.rob_id))
    if (unread.size() != 0 || unwritten.size() != 0)
      `uvm_fatal("RESP", $sformatf(
                 "response before all traffic: %0d rows unread, %0d unwritten",
                 unread.size(),
                 unwritten.size()
                 ))
    resp_count++;
  endfunction

  function bit done();
    return stim != null && cmd_count == 1 && resp_count == 1;
  endfunction

  function void reset_counters();
    stim = null;
    unread.delete();
    unwritten.delete();
    kind.delete();
    cmd_count  = 0;
    resp_count = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d resp=%0d", cmd_count, resp_count))
  endfunction
endclass
