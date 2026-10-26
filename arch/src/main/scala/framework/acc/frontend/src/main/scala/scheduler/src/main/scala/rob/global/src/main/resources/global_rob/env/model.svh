class model extends uvm_subscriber #(sample_item);
  `uvm_component_utils(model)
  localparam int N = `GROB_ENTRIES, V = `GROB_VBANKS, A = `GROB_ALIASES;
  transaction slots[N], order[$], completion_queue[$], active_completion;
  int current[V], tail = 0, cycle = 0, serial = 0;
  longint unsigned next_inst = 1;
  bit [A-1:0] free_aliases, busy;
  bit hold_issue = 0, hold_mapping = 0, hold_complete = 0, reverse_complete = 0;
  int allocated = 0, selected = 0, delivered = 0, completed = 0, retired = 0, reordered = 0;
  sample_item latest;
  int cancelled_live = 0;
  int trace_offsets[4][10] = `GROB_TRACE_OFFSETS;
  int query_offsets[N][9] = `GROB_QUERY_OFFSETS;
  int ba_offsets[9] = `GROB_BA_OFFSETS;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (
      int count, bit stalled, bit mapping_stall, bit reordered, bit config_op, bit movement
  );
    option.per_instance = 1;
    occupancy: coverpoint count {bins empty = {0}; bins partial = {[1 : N - 1]}; bins full = {N};}
    issue_backpressure: coverpoint stalled;
    mapping_backpressure: coverpoint mapping_stall;
    out_of_order_completion: coverpoint reordered;
    configuration: coverpoint config_op;
    movement_command: coverpoint movement;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    behavior = new();
  endfunction
  function void predict_bat(logic [`GROB_INPUT_BITS-1:0] pins, output bank_access_t renamed,
                            output bit ready, rename, reset_read, output bit [1:0] stale_valid,
                            output int stale0, stale1);
    int fresh, wr, r0, r1, rw, funct;
    bit update, resets_write;
    renamed = '0;
    fresh   = 0;
    for (int index = 0; index < A; index++)
    if (free_aliases[index]) begin
      fresh = index;
      break;
    end
    wr = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_WR_BANK_ID);
    r0 = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_BANK_0_ID);
    r1 = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_BANK_1_ID);
    rw = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_WR_ID);
    funct = `GI(pins, ALLOC_BITS_OPERATION_CMD_RAW_INST) >> 25;
    update = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_WR_UPDATE);
    rename = `GROB_RENAMING &&
    `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_WR_BANK_VALID)
    && wr < V && funct != 32 && !update;
    reset_read = funct == 32 && (`GI(pins, ALLOC_BITS_OPERATION_CMD_RS2DATA) & 64'h1000) != 0;
    ready = !rename || free_aliases != 0;
    `BA(renamed, RD_BANK_0_VALID) = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_BANK_0_VALID);
    `BA(renamed, RD_BANK_1_VALID) = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_BANK_1_VALID);
    `BA(renamed, RD_WR_VALID) = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_RD_WR_VALID);
    `BA(renamed, WR_BANK_VALID) = `GI(pins, ALLOC_BITS_OPERATION_BANKACCESS_WR_BANK_VALID);
    `BA(renamed, WR_UPDATE) = update;
    `BA(renamed, RD_BANK_0_ID) = `BA(renamed, RD_BANK_0_VALID) ? current[r0] : 0;
    `BA(renamed, RD_BANK_1_ID) = `BA(renamed, RD_BANK_1_VALID) ? current[r1] : 0;
    `BA(renamed, RD_WR_ID) = `BA(renamed, RD_WR_VALID) ? current[rw] : 0;
    `BA(renamed, WR_BANK_ID) = rename ? fresh : update ? current[wr] : wr;
    resets_write = `BA(renamed, WR_BANK_VALID) && !rename && !update && current[wr] != wr;
    stale_valid[0] = rename || (!reset_read && resets_write) || (reset_read && current[r0] != r0);
    stale_valid[1] = reset_read && resets_write && r0 != wr;
    stale0 = reset_read ? current[r0] : current[wr];
    stale1 = current[wr];
  endfunction
  function bit hazard(bank_access_t access);
    return (
    `BA(access, RD_BANK_0_VALID)
    && busy[
    `BA(access, RD_BANK_0_ID)
    ]) || (
    `BA(access, RD_BANK_1_VALID)
    && busy[
    `BA(access, RD_BANK_1_ID)
    ]) || (
    `BA(access, RD_WR_VALID)
    && busy[
    `BA(access, RD_WR_ID)
    ]);
  endfunction
  function bit eligible(transaction candidate);
    bit conflict, overlap;
    if (hazard(candidate.renamed)) return 0;
    foreach (order[i]) begin
      transaction older = order[i];
      if (older == candidate) break;
      overlap=((candidate.reads&older.writes)|(candidate.writes&(older.reads|older.writes)))!=0;
      conflict = (!`GROB_RENAMING || !`GROB_LAYOUT) ? overlap : 0;
      conflict |= (
      `OP(candidate.operation, BANKACCESS_WR_UPDATE)
      && overlap) || (
      `OP(older.operation, BANKACCESS_WR_UPDATE)
      && (candidate.reads & older.writes) != 0);
      if (!older.completed && conflict) return 0;
      if((candidate.config_op&&older.config_op)||((candidate.config_op||older.config_op)&&
          ((candidate.reads|candidate.writes)&(older.reads|older.writes))!=0))
        return 0;
      if (candidate.movement && !older.completed) return 0;
    end
    return 1;
  endfunction
  function void verify_trace(logic [`GROB_OUTPUT_BITS-1:0] outputs, int index, transaction r,
                             int kind);
    evidence::check(
        outputs[trace_offsets[index][0]+:8]==kind&&
      outputs[trace_offsets[index][1]+:64]==`GROB_OWNER&&outputs[trace_offsets[index][2]+:32]==r.slot&&
      outputs[trace_offsets[index][3]+:64]==r.inst_id&&outputs[trace_offsets[index][4]+:32]==
        `OP(r.operation, DOMAIN_ID)
        && outputs[trace_offsets[index][5]+:32] == r.funct &&
            outputs[trace_offsets[index][6]+:64] == r.pc && outputs[trace_offsets[index][7]+:64] ==
        `OP(r.operation, CMD_RS1DATA) && outputs[trace_offsets[index][8]+:64] ==
        `OP(r.operation, CMD_RS2DATA),
        "Global ROB public instruction observation changed its fields");
  endfunction
  function void write(sample_item t);
    transaction candidate, map_retire, r, ignored, oldest_incomplete;
    result_item expected, actual;
    bank_access_t renamed, query_access;
    bit bat_ready, rename, reset_read, input_fire, select_fire, issue_fire, complete_fire, map_fire;
    bit [1:0] stale_valid;
    bit movement_live, has_retirement, needs_mapping, reads_previous, completion_reordered;
    int stale0, stale1, head, map_index, retire_count, cid;
    latest = t;
    if (t.reset) begin
      reset_ap.write(1);
      cancelled_live += order.size();
      order.delete();
      completion_queue.delete();
      active_completion = null;
      tail = 0;
      next_inst = 1;
      busy = 0;
      free_aliases = '1;
      foreach (slots[i]) slots[i] = null;
      foreach (current[i]) begin
        current[i] = i;
        free_aliases[i] = 0;
      end
      return;
    end
    cycle++;
    completion_reordered = 0;
    predict_bat(t.inputs, renamed, bat_ready, rename, reset_read, stale_valid, stale0, stale1);
    head = order.size() ? order[0].slot : tail;
    input_fire = `GI(t.inputs, ALLOC_VALID) && `GO(t.outputs, ALLOC_READY);
    select_fire = `GO(t.outputs, SCOREBOARD_ISSUE_VALID);
    issue_fire = `GO(t.outputs, ISSUE_VALID) && `GI(t.inputs, ISSUE_READY);
    complete_fire = `GI(t.inputs, COMPLETE_VALID) && `GO(t.outputs, COMPLETE_READY);
    map_fire = `GO(t.outputs, MAPPING_VALID) && `GI(t.inputs, MAPPING_READY);
    candidate = null;
    map_retire = null;
    map_index = 0;
    retire_count = 0;
    movement_live = 0;
    foreach (order[i]) begin
      if (candidate == null && !order[i].selected && !order[i].completed) candidate = order[i];
      if (order[i].movement && !order[i].completed) movement_live = 1;
    end
    // Retirement is a completed FIFO prefix. A final mapping acknowledgement releases its entry.
    foreach (order[i]) begin
      if (!order[i].completed) break;
      if (order[i].stale_valid != 0) begin
        map_retire = order[i];
        map_index  = order[i].stale_valid[0] ? 0 : 1;
        break;
      end
    end
    if (map_fire && `GO(t.outputs, MAPPING_BITS_RETIRE)) begin
      evidence::check(map_retire != null, "Global ROB retired an unsolicited mapping");
      evidence::check(`GO(t.outputs, MAPPING_BITS_BANK)
                      == (map_index == 0 ? map_retire.stale_raw : map_retire.raw_write) &&
                      `GO(t.outputs, MAPPING_BITS_VBANK)
                      == (map_index == 0 ? map_retire.new_alias : map_retire.raw_write) &&
                      `GO(t.outputs, MAPPING_BITS_PREVIOUS) == map_retire.stale[map_index] &&
                      `GO(t.outputs, MAPPING_BITS_THREADID) == `GROB_OWNER && !
                      `GO(t.outputs, MAPPING_BITS_ALLOCATE) && !`GO(t.outputs, MAPPING_BITS_REUSE)
                      && !`GO(t.outputs, MAPPING_BITS_WRITE),
                      "Global ROB mapping retirement metadata mismatch");
    end
    foreach (order[i]) begin
      if (!order[i].completed) break;
      if (order[i].stale_valid != 0 && !(order[i] == map_retire && map_fire &&
          `GO(t.outputs, MAPPING_BITS_RETIRE)
          && $countones(
              order[i].stale_valid
          ) == 1))
        break;
      retire_count++;
    end
    has_retirement = retire_count != 0;
    evidence::check(`GO(t.outputs, EMPTY) == (order.size() == 0) && `GO(t.outputs, FULL)
                    == (order.size() == N), "Global ROB occupancy flags mismatch");
    evidence::check(`GO(t.outputs, ALLOC_READY)
                    == (order.size() < N && !has_retirement && bat_ready && !movement_live),
                    "Global ROB admission rule mismatch");
    evidence::check(`GO(t.outputs, BAT_ALLOC_VALID) == input_fire,
                    "Global ROB BAT allocation handshake mismatch");
    evidence::check(`GO(t.outputs, TRACEINSTRUCTIONS_0_ENABLE) == input_fire &&
                    `GO(t.outputs, TRACEINSTRUCTIONS_2_ENABLE) == complete_fire,
                    "Global ROB public observation handshakes mismatch");
    evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_VALID) == (input_fire && rename),
                    "Global ROB scoreboard allocation handshake mismatch");
    evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_VALID) == complete_fire,
                    "Global ROB completion handshake mismatch");
    evidence::check(`GO(t.outputs, TRACEINSTRUCTIONS_1_ENABLE) == select_fire,
                    "Global ROB issue observation mismatch");
    for (int age = 0; age < N; age++) begin
      int slot = (head + age) % N;
      if (slots[slot] != null)
        for (int field = 0; field < 9; field++) begin
          if (field % 2 == 0)
            evidence::check(
                t.outputs[query_offsets[age][field]] === slots[slot].renamed[ba_offsets[field]],
                "Global ROB query flag lost its entry");
          else
            evidence::check(
                t.outputs[query_offsets[age][field]+:`GROB_BANK_BITS]===slots[slot].renamed[ba_offsets[field]+:`GROB_BANK_BITS],
                "Global ROB query ID lost its alias");
        end
    end
    if (map_fire && !`GO(t.outputs, MAPPING_BITS_RETIRE)) begin
      evidence::check(candidate != null && eligible(candidate) && select_fire,
                      "Global ROB mapping allocation bypassed a dependency");
      needs_mapping = candidate.allocated || (
      `BA(candidate.renamed, WR_BANK_VALID)
      && !
      `BA(candidate.renamed, WR_UPDATE)
      && candidate.raw_write < V && !candidate.config_op);
      reads_previous = (
      `BA(candidate.renamed, RD_BANK_0_VALID)
      &&
      `BA(candidate.renamed, RD_BANK_0_ID)
      == candidate.stale[0]) || (
      `BA(candidate.renamed, RD_BANK_1_VALID)
      &&
      `BA(candidate.renamed, RD_BANK_1_ID)
      == candidate.stale[0]) || (
      `BA(candidate.renamed, RD_WR_VALID)
      &&
      `BA(candidate.renamed, RD_WR_ID)
      == candidate.stale[0]);
      evidence::check(needs_mapping && `GO(t.outputs, MAPPING_BITS_BANK) == candidate.raw_write &&
                      `GO(t.outputs, MAPPING_BITS_VBANK) == `BA(candidate.renamed, WR_BANK_ID) &&
                      `GO(t.outputs, MAPPING_BITS_PREVIOUS) == candidate.stale[0] &&
                      `GO(t.outputs, MAPPING_BITS_THREADID) == `GROB_OWNER &&
                      `GO(t.outputs, MAPPING_BITS_ALLOCATE) == candidate.allocated &&
                      `GO(t.outputs, MAPPING_BITS_REUSE)
                      == (candidate.allocated && candidate.slot == head && !reads_previous) &&
                      `GO(t.outputs, MAPPING_BITS_WRITE) == (`BA(candidate.renamed, WR_BANK_VALID)
                      && !`BA(candidate.renamed, WR_UPDATE)
                      && candidate.raw_write < V && !candidate.config_op),
                      "Global ROB mapping issue metadata mismatch");
    end
    if (select_fire) begin
      evidence::check(candidate != null && eligible(candidate),
                      "Global ROB selected a blocked or younger instruction");
      needs_mapping = candidate.allocated || (
      `BA(candidate.renamed, WR_BANK_VALID)
      && !
      `BA(candidate.renamed, WR_UPDATE)
      && candidate.raw_write < V && !candidate.config_op);
      if (needs_mapping)
        evidence::check(map_fire && !`GO(t.outputs, MAPPING_BITS_RETIRE),
                        "Global ROB selected work before mapping acceptance");

      evidence::check(`GO(t.outputs, TRACEINSTRUCTIONS_1_ROB_ID) == candidate.slot &&
                      `GO(t.outputs, TRACEINSTRUCTIONS_1_PC) == candidate.pc &&
                      `GO(t.outputs, TRACEINSTRUCTIONS_1_INST_ID) == candidate.inst_id &&
                      `GO(t.outputs, TRACEINSTRUCTIONS_1_THREAD_ID) == `GROB_OWNER,
                      "Global ROB issue identity mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_BANK_0_VALID) ===
                      `BA(candidate.renamed, RD_BANK_0_VALID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_BANK_0_ID) ===
                      `BA(candidate.renamed, RD_BANK_0_ID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_BANK_1_VALID) ===
                      `BA(candidate.renamed, RD_BANK_1_VALID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_BANK_1_ID) ===
                      `BA(candidate.renamed, RD_BANK_1_ID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_WR_VALID) ===
                      `BA(candidate.renamed, RD_WR_VALID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_RD_WR_ID) ===
                      `BA(candidate.renamed, RD_WR_ID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_WR_BANK_VALID) ===
                      `BA(candidate.renamed, WR_BANK_VALID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_WR_BANK_ID) ===
                      `BA(candidate.renamed, WR_BANK_ID),
                      "Global ROB scoreboard issue field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_ISSUE_BITS_WR_UPDATE) ===
                      `BA(candidate.renamed, WR_UPDATE),
                      "Global ROB scoreboard issue field mismatch");
      verify_trace(t.outputs, 1, candidate, 1);
      candidate.selected = 1;
      selected++;
    end
    if (issue_fire) begin
      int slot = `GO(t.outputs, ISSUE_BITS_ROB_ID);
      r = slots[slot];
      evidence::check(r != null && r.selected && !r.delivered,
                      "Global ROB delivered an unselected or duplicate instruction");
      actual = result_item::type_id::create("issued");
      actual.payload = t.outputs[`GROB_ISSUE_OFFSET+:`GROB_ISSUE_BITS];
      actual_ap.write(actual);
      r.delivered = 1;
      r.due = cycle + (r.serial % 4 == 0 ? 17 : 3);
      completion_queue.push_back(r);
      delivered++;
    end
    if (complete_fire) begin
      cid = `GI(t.inputs, COMPLETE_BITS);
      r   = slots[cid];
      evidence::check(r != null && r.delivered && !r.completed && active_completion == r,
                      "Global ROB accepted an invalid completion");
      oldest_incomplete = null;
      foreach (order[i])
      if (oldest_incomplete == null && !order[i].completed) oldest_incomplete = order[i];
      if (r != oldest_incomplete) begin
        reordered++;
        completion_reordered = 1;
      end
      evidence::check(`GO(t.outputs, TRACEINSTRUCTIONS_2_ROB_ID) == r.slot &&
                      `GO(t.outputs, TRACEINSTRUCTIONS_2_PC) == r.pc &&
                      `GO(t.outputs, TRACEINSTRUCTIONS_2_INST_ID) == r.inst_id,
                      "Global ROB completion identity mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_BANK_0_VALID) ===
                      `BA(r.renamed, RD_BANK_0_VALID),
                      "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_BANK_0_ID) ===
                      `BA(r.renamed, RD_BANK_0_ID),
                      "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_BANK_1_VALID) ===
                      `BA(r.renamed, RD_BANK_1_VALID),
                      "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_BANK_1_ID) ===
                      `BA(r.renamed, RD_BANK_1_ID),
                      "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_WR_VALID) ===
                      `BA(r.renamed, RD_WR_VALID), "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_RD_WR_ID) ===
                      `BA(r.renamed, RD_WR_ID), "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_WR_BANK_VALID) ===
                      `BA(r.renamed, WR_BANK_VALID),
                      "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_WR_BANK_ID) ===
                      `BA(r.renamed, WR_BANK_ID), "Global ROB scoreboard complete field mismatch");
      evidence::check(`GO(t.outputs, SCOREBOARD_COMPLETE_BITS_WR_UPDATE) ===
                      `BA(r.renamed, WR_UPDATE), "Global ROB scoreboard complete field mismatch");
      verify_trace(t.outputs, 2, r, 0);
      r.completed = 1;
      completed++;
      active_completion = null;
    end
    if (input_fire) begin
      evidence::check(t.inputs[`GROB_IN_RENAMED_OFFSET+:`GROB_IN_RENAMED_BITS] === renamed,
                      "Global ROB peer alias changed before allocation");
      evidence::check(`GO(t.outputs, BAT_ALLOC_RENAME) == rename &&
                      `GO(t.outputs, BAT_ALLOC_RESET_READ0) == reset_read,
                      "Global ROB requested the wrong alias operation");
      r = new;
      r.operation = t.inputs[`GROB_IN_OPERATION_OFFSET+:`GROB_IN_OPERATION_BITS];
      r.renamed = renamed;
      r.slot = tail;
      r.serial = serial++;
      r.pc = `OP(r.operation, CMD_PC);
      r.inst_id = `OP(r.operation, RUNTIME) ? next_inst++ : 0;
      r.funct = `OP(r.operation, CMD_RAW_INST) >> 25;
      r.config_op = r.funct == 32;
      r.movement = r.funct == 13;
      r.raw_write = `OP(r.operation, BANKACCESS_WR_BANK_ID);
      r.new_alias = reset_read ?
      `OP(r.operation, BANKACCESS_RD_BANK_0_ID)
      :
      `BA(renamed, WR_BANK_ID);
      r.stale_raw = reset_read ? `OP(r.operation, BANKACCESS_RD_BANK_0_ID) : r.raw_write;
      r.allocated = rename;
      r.stale_valid = stale_valid;
      r.stale[0] = stale0;
      r.stale[1] = stale1;
      r.reads = 0;
      r.writes = 0;
      if (`OP(r.operation, BANKACCESS_RD_BANK_0_VALID))
        r.reads[`OP(r.operation, BANKACCESS_RD_BANK_0_ID)] = 1;
      if (`OP(r.operation, BANKACCESS_RD_BANK_1_VALID))
        r.reads[`OP(r.operation, BANKACCESS_RD_BANK_1_ID)] = 1;
      if (`OP(r.operation, BANKACCESS_RD_WR_VALID))
        r.reads[`OP(r.operation, BANKACCESS_RD_WR_ID)] = 1;
      if (`OP(r.operation, BANKACCESS_WR_BANK_VALID)) r.writes[r.raw_write] = 1;
      r.expected = '0;
      r.expected[`GROB_OUT_COMMAND_OFFSET-`GROB_ISSUE_OFFSET+:`GROB_OUT_COMMAND_BITS] = r.operation;
      r.expected[`GROB_OUT_CONTEXT_OFFSET-`GROB_ISSUE_OFFSET+:`GROB_OUT_CONTEXT_BITS]=t.inputs[`GROB_IN_CONTEXT_OFFSET+:`GROB_IN_CONTEXT_BITS];
      r.expected[`GROB_OUT_RENAMED_OFFSET-`GROB_ISSUE_OFFSET+:`GROB_OUT_RENAMED_BITS] = renamed;
      `ISS(r.expected, ROB_ID) = tail;
      `ISS(r.expected, INSTID) = r.inst_id;
      expected = result_item::type_id::create("allocated");
      expected.payload = r.expected;
      expected_ap.write(expected);
      verify_trace(t.outputs, 0, r, 2);
      slots[tail] = r;
      order.push_back(r);
      tail = (tail + 1) % N;
      allocated++;
      if (rename) begin
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_BANK_0_VALID) ===
                        `BA(renamed, RD_BANK_0_VALID),
                        "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_BANK_0_ID) ===
                        `BA(renamed, RD_BANK_0_ID), "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_BANK_1_VALID) ===
                        `BA(renamed, RD_BANK_1_VALID),
                        "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_BANK_1_ID) ===
                        `BA(renamed, RD_BANK_1_ID), "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_WR_VALID) ===
                        `BA(renamed, RD_WR_VALID), "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_RD_WR_ID) === `BA(renamed, RD_WR_ID),
                        "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_WR_BANK_VALID) ===
                        `BA(renamed, WR_BANK_VALID), "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_WR_BANK_ID) ===
                        `BA(renamed, WR_BANK_ID), "Global ROB scoreboard alloc field mismatch");
        evidence::check(`GO(t.outputs, SCOREBOARD_ALLOC_BITS_WR_UPDATE) === `BA(renamed, WR_UPDATE),
                        "Global ROB scoreboard alloc field mismatch");
      end
      if (rename) free_aliases[`BA(renamed, WR_BANK_ID)] = 0;
      if (`BA(renamed, WR_BANK_VALID)) current[r.raw_write] = `BA(renamed, WR_BANK_ID);
      if (reset_read)
        current[
        `OP(r.operation, BANKACCESS_RD_BANK_0_ID)
        ] =
        `OP(r.operation, BANKACCESS_RD_BANK_0_ID);
    end
    if (select_fire && `BA(candidate.renamed, WR_BANK_VALID))
      busy[`BA(candidate.renamed, WR_BANK_ID)] = 1;
    if (input_fire && rename) busy[`BA(renamed, WR_BANK_ID)] = 1;
    if (complete_fire) begin
      r = slots[cid];
      if (`BA(r.renamed, WR_BANK_VALID)) busy[`BA(r.renamed, WR_BANK_ID)] = 0;
    end
    if (map_fire && `GO(t.outputs, MAPPING_BITS_RETIRE)) begin
      map_retire.stale_valid[map_index] = 0;
      if (map_retire.stale[map_index] >= V) free_aliases[map_retire.stale[map_index]] = 1;
    end
    behavior.sample(order.size(), `GO(t.outputs, ISSUE_VALID) && !`GI(t.inputs, ISSUE_READY),
                    `GO(t.outputs, MAPPING_VALID) && !`GI(t.inputs, MAPPING_READY),
                    completion_reordered, input_fire && (
                    `GI(t.inputs, ALLOC_BITS_OPERATION_CMD_RAW_INST) >> 25) == 32, input_fire && (
                    `GI(t.inputs, ALLOC_BITS_OPERATION_CMD_RAW_INST) >> 25) == 13);
    repeat (retire_count) begin
      ignored = order.pop_front();
      slots[ignored.slot] = null;
      retired++;
    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    evidence::check(order.size() == 0 && completion_queue.size() == 0 && active_completion == null,
                    "Global ROB left live instructions");
    evidence::check(retired + cancelled_live == allocated,
                    "Global ROB allocation/retirement accounting mismatch");
    evidence::check(reordered != 0, "Global ROB did not exercise out-of-order completion");
  endfunction
endclass
