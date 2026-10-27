class trace_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(trace_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, trace_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, trace_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, trace_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, trace_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, trace_scoreboard) resp_imp;

  bb_blink_cmd_item stim_q[$];
  bb_blink_cmd_item issued_q[$];
  int unsigned expected_cmds;
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
    bb_blink_cmd_item want;
    // Compare as plain BLink commands; the comparer rejects differing types.
    want = bb_blink_cmd_item::type_id::create("want");
    want.copy(item);
    stim_q.push_back(want);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    bb_blink_cmd_item want;
    if (stim_q.size() == 0) `uvm_fatal("CMD", "command observed before stimulus")
    if (issued_q.size() != 0) `uvm_fatal("CMD", "command accepted before the previous response")
    want = stim_q.pop_front();
    verification_pkg::evidence::checks++;
    if (!item.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", item.sprint(), want.sprint()))
    issued_q.push_back(want);
    cmd_count++;
  endfunction

  function void write_read(bb_blink_read_item item);
    `uvm_fatal("READ", $sformatf("TraceBall touches no bank, got a read of bank %0d", item.bank_id))
  endfunction

  function void write_write(bb_blink_write_item item);
    `uvm_fatal("WRITE", $sformatf("TraceBall touches no bank, got a write of bank %0d", item.bank_id
               ))
  endfunction

  // The reference model in the DPI library checks each trace record as the
  // RTL emits it; the record count is checked once the case has drained.
  function void write_resp(bb_blink_resp_item item);
    bb_blink_cmd_item want;
    int records;
    if (issued_q.size() == 0) `uvm_fatal("RESP", "response without a command")
    want = issued_q.pop_front();
    if (item.rob_id !== want.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, want.rob_id))
    resp_count++;
    records = trace_case_records();
    verification_pkg::evidence::checks++;
    if (records < 0) `uvm_fatal("TRACE", "trace record mismatch, see [TRACE-REF]")
  endfunction

  function bit done();
    return expected_cmds != 0 && cmd_count == expected_cmds && resp_count == expected_cmds;
  endfunction

  function void reset_counters(int unsigned num_cmds);
    stim_q.delete();
    issued_q.delete();
    expected_cmds = num_cmds;
    cmd_count     = 0;
    resp_count    = 0;
  endfunction

  function void check_records();
    int records;
    records = trace_case_records();
    verification_pkg::evidence::checks++;
    if (records < 0) `uvm_fatal("TRACE", "trace record mismatch, see [TRACE-REF]")
    if (records != expected_cmds)
      `uvm_fatal("TRACE", $sformatf("%0d trace records for %0d commands", records, expected_cmds))
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done())
      `uvm_fatal("SCB", $sformatf(
                 "incomplete: cmds=%0d resp=%0d of %0d", cmd_count, resp_count, expected_cmds))
    check_records();
  endfunction
endclass
