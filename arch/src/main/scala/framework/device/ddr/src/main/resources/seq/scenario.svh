task submit(int client, int id, u64 address, bit write, bit [63:0] mask = '1);
  req_t q = '0;
  q[`DF(REQ, ID)] = id;
  q[`DF(REQ, ADDR)] = address;
  q[`DF(REQ, WRITE)] = write;
  q[`DF(REQ, DATA)] = pattern(id + client * 17 + int'(address >> 6));
  q[`DF(REQ, MASK)] = mask;
  @(negedge ctl.clock);
  req[client].bits  = q;
  req[client].valid = 1;
  do @(req[client].sample); while (!req[client].sample.ready);
  @(negedge ctl.clock);
  req[client].valid = 0;
endtask
task batch(u64 base, bit write);
  for (int c = 0; c < 2; c++)
    for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      initialize(base + c * 4096 + i * 64, i + c * 7 + int'(base >> 6));
  fork
    begin
      for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      submit(0, 12'h800 + i, base + i * 64, write,
             i == 0 ? 0 : i == 1 ? '1 : i == 2 ? 64'h5555555555555555 : 64'haaaaaaaaaaaaaaaa);
    end
    begin
      for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      submit(1, 12'h800 + i, base + 4096 + i * 64, write);
    end
  join
endtask
task drain();
  wait (commands.num() == 0);
  @(ctl.sample);
  verify_contract(
      !reading_ids.num()&&!writing_ids.num()&&!aw_ids.size()&&!w_lines.size()&&!assembling_beat&&!r_active&&!b_active,
      "Backend did not drain");
endtask
task joint_reset();
  @(negedge ctl.clock);
  ctl.reset = 1;
  req[0].valid = 0;
  req[1].valid = 0;
  repeat (3) @(ctl.sample);
  behavior.sample(0, 0, 0, 0, commands.num(), commands.num() != 0);
  cancelled += commands.num();
  commands.delete();
  reading_ids.delete();
  writing_ids.delete();
  aw_ids.delete();
  aw_keys.delete();
  aw_serials.delete();
  w_lines.delete();
  scoreboard.cancel_pending();
  assembling = '0;
  assembling_beat = 0;
  r_active = 0;
  b_active = 0;
  @(negedge ctl.clock);
  ctl.reset = 0;
endtask
function int finished_commands();
  int count = 0;
  foreach (commands[key]) if (commands[key].done) count++;
  return count;
endfunction
task invalid_scenario();
  int id, beats;
  bit aw_seen;
  if (invalid_case == 0) begin
    submit(0, 0, 'h10001, 0);
  end else if (invalid_case == 1) begin
    @(negedge ctl.clock);
    r.bits = '0;
    r.bits[`DF(R, ID)] = 0;
    r.valid = 1;
  end else if (invalid_case == 2 || invalid_case == 6) begin
    aw.ready = 1;
    w.ready  = invalid_case == 6;
    submit(0, 0, 'h10000, 1);
    beats   = 0;
    aw_seen = 0;
    do begin
      @(ctl.sample);
      if (aw.sample.valid && aw.sample.ready) begin
        id = aw.sample.bits[`DF(A, ID)];
        aw_seen = 1;
      end
      if (w.sample.valid && w.sample.ready) beats++;
    end while (!aw_seen || (invalid_case == 6 && beats != `DDR_BEATS));
    @(negedge ctl.clock);
    b.bits = '0;
    b.bits[`DF(B, ID)] = id;
    b.bits[`DF(B, RESP)] = invalid_case == 6 ? 1 : 0;
    b.valid = 1;
  end else begin
    ar.ready = 1;
    submit(0, 0, 'h10000, 0);
    do @(ctl.sample); while (!ar.sample.valid || !ar.sample.ready);
    id = ar.sample.bits[`DF(A, ID)];
    for (int beat = 0; beat < (invalid_case == 4 ? `DDR_BEATS : 1); beat++) begin
      @(negedge ctl.clock);
      r.bits = '0;
      r.bits[`DF(R, ID)] = id;
      r.bits[`DF(R, LAST)] = invalid_case == 3;
      r.bits[`DF(R, RESP)] = invalid_case == 5 ? 1 : 0;
      r.valid = 1;
      do @(r.sample); while (!r.sample.ready);
    end
  end
  repeat (8) @(ctl.sample);
  `uvm_fatal("MISSING_ASSERTION", "Invalid protocol did not trigger expected DUT assertion")
endtask
task execute();
  int before_aw, before_w, before_ar, before_r;
  int initial_returned;
  ctl.reset = 1;
  req[0].valid = 0;
  req[1].valid = 0;
  req[0].bits = '0;
  req[1].bits = '0;
  resp[0].ready = 0;
  resp[1].ready = 0;
  aw.ready = 0;
  ar.ready = 0;
  w.ready = 0;
  b.valid = 0;
  b.bits = '0;
  r.valid = 0;
  r.bits = '0;
  repeat (5) @(negedge ctl.clock);
  ctl.reset = 0;
  if (invalid_case >= 0) begin
    invalid_scenario();
    return;
  end
  fork
    service();
  join_none
  // W must progress even when AW has never been accepted; all eight slots stay live until B.
  hold_aw = 1;
  hold_b = 1;
  hold_resp[0] = 1;
  batch('h10000, 1);
  wait (w_bursts == `DDR_SLOTS);
  verify_contract(aw_count == 0 && returned[0] == 0 && returned[1] == 0,
                  "W-before-AW or B-delayed behavior failed");
  hold_aw = 0;
  wait (aw_count == `DDR_SLOTS);
  initialize('h18000, 99);
  fork
    submit(0, 12'h800, 'h18000, 1);
    begin
      hold_b = 0;
      wait (returned[1] == `DDR_SLOTS_PER_CLIENT);
      verify_contract(returned[0] == 0, "Client0 backpressure leaked");
      hold_resp[0] = 0;
    end
  join
  drain();
  for (int c = 0; c < 2; c++)
    for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      submit(c, 12'h900 + i, 'h10000 + c * 4096 + i * 64, 0);
  drain();
  // Conversely allow addresses to run ahead of every W beat.
  before_aw = aw_count;
  before_w = w_bursts;
  hold_w = 1;
  batch('h20000, 1);
  wait (aw_count == before_aw + `DDR_SLOTS);
  verify_contract(w_bursts == before_w, "AW-before-W stalled ordering failed");
  hold_w = 0;
  drain();
  // Concurrent reads and writes: hold B until all independent reads return.
  hold_b = 1;
  initial_returned = returned[0] + returned[1];
  for (int c = 0; c < 2; c++)
    for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++) initialize('h30000 + c * 4096 + i * 64, 30 + i);
  fork
    begin
      for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      submit(0, 12'ha00 + i, 'h30000 + i * 64, i[0]);
    end
    begin
      for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
      submit(1, 12'ha00 + i, 'h31000 + i * 64, i[0]);
    end
  join
  wait (returned[0] + returned[1] == initial_returned + 2 * ((`DDR_SLOTS_PER_CLIENT + 1) / 2));
  hold_b = 0;
  drain();
  for (int i = 0; i < 4; i++) begin
    initialize('h40000 + i * 64, 40 + i);
    read_error['h40000+i*64] = i % `DDR_BEATS;
    submit(i % 2, 12'hb00 + i, 'h40000 + i * 64, 0);
  end
  initialize('h41000, 44);
  write_error['h41000] = 2;
  submit(1, 12'hb10, 'h41000, 1);
  initialize('h41040, 45);
  write_error['h41040] = 3;
  submit(0, 12'hb10, 'h41040, 1);
  drain();
  before_ar = ar_count;
  hold_r = 1;
  batch('h48000, 0);
  wait (ar_count == before_ar + `DDR_SLOTS);
  r_cursor = `DDR_ID_BASE;
  hold_r   = 0;
  drain();
  // Each byte strobe is independently checked, then read back through the reference.
  for (int i = 0; i < 64; i++) begin
    initialize('h50000 + i * 64, 50 + i);
    submit(i % 2, 12'hc00 + i, 'h50000 + i * 64, 1, 64'h1 << i);
    drain();
    submit(i % 2, 12'hd00 + i, 'h50000 + i * 64, 0);
    drain();
  end
  // Coordinated reset cancels queued AW-only, W-only and stalled line completions.
  before_aw = aw_count;
  hold_w = 1;
  batch('h60000, 1);
  wait (aw_count == before_aw + `DDR_SLOTS);
  joint_reset();
  hold_w   = 0;
  before_w = w_bursts;
  hold_aw  = 1;
  batch('h70000, 1);
  wait (w_bursts == before_w + `DDR_SLOTS);
  joint_reset();
  hold_aw = 0;
  before_ar = ar_count;
  before_r = r_beats;
  hold_r = 1;
  batch('h78000, 0);
  wait (ar_count == before_ar + `DDR_SLOTS);
  hold_r = 0;
  wait (r_beats >= before_r + 1);
  joint_reset();
  hold_resp[0] = 1;
  hold_resp[1] = 1;
  batch('h80000, 0);
  do @(ctl.sample); while (finished_commands() != `DDR_SLOTS);
  joint_reset();
  hold_resp[0] = 0;
  hold_resp[1] = 0;
  batch('h90000, 0);
  drain();
  verify_contract(
      peak==`DDR_SLOTS&&(`DDR_BEATS==1?interleaves==0:interleaves>0)&&b_reorders>0&&r_reorders>0&&request_stalls>0&&response_stalls>0&&aw_stalls>0&&w_stalls>0&&error_results==6&&cancelled==4*`DDR_SLOTS,
      "Missing concurrency, stalls, errors or reset coverage");
  verify_contract(accepted == returned[0] + returned[1] + cancelled, "Lost line response");
  `uvm_info(
      "DDR_PASS",
      $sformatf(
          "accepted=%0d returned=%0d/%0d cancelled=%0d peak=%0d errors=%0d readInterleave=%0d R/Breorder=%0d/%0d stalls req/resp/AW/W=%0d/%0d/%0d/%0d; full64B payload/mask/tag and all queues drained",
          accepted, returned[0], returned[1], cancelled, peak, error_results, interleaves,
          r_reorders, b_reorders, request_stalls, response_stalls, aw_stalls, w_stalls), UVM_LOW)
endtask
