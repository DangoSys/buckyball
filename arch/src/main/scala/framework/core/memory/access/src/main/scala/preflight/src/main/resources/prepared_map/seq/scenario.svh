task reserve_tag(int id);
  @(negedge ctl.clock);
  reserve.bits = '0;
  reserve.bits[`PF(RESERVE, ID)] = id;
  reserve.valid = 1;
  do @(ctl.sample); while (!reserve.sample.ready);
  @(negedge ctl.clock);
  reserve.valid = 0;
endtask
task record(int id, u64 va, u64 pa, int bytes, bit write, bit last, int error = 0);
  @(negedge ctl.clock);
  prepared.bits = '0;
  prepared.bits[`PF(OUT, ID)] = id;
  prepared.bits[`PF(OUT, VA)] = va;
  prepared.bits[`PF(OUT, PA)] = pa;
  prepared.bits[`PF(OUT, BYTES)] = bytes;
  prepared.bits[`PF(OUT, WRITE)] = write;
  prepared.bits[`PF(OUT, LAST)] = last;
  prepared.bits[`PF(OUT, ERROR)] = error;
  prepared.valid = 1;
  do @(ctl.sample); while (!prepared.sample.ready);
  @(negedge ctl.clock);
  prepared.valid = 0;
endtask
function logic [`PF_QUERY_WIDTH-1:0] query(int id, u64 va, int bytes, bit write);
  logic [`PF_QUERY_WIDTH-1:0] q = '0;
  q[`PF(QUERY, VALID)] = 1;
  q[`PF(QUERY, ID)] = id;
  q[`PF(QUERY, VA)] = va;
  q[`PF(QUERY, BYTES)] = bytes;
  q[`PF(QUERY, WRITE)] = write;
  return q;
endfunction
task probes(logic [`PF_QUERY_WIDTH-1:0] a, logic [`PF_QUERY_WIDTH-1:0] b, int clocks = 3);
  @(negedge ctl.clock);
  ctl.query_0 = a;
  ctl.query_1 = b;
  repeat (clocks) @(ctl.sample);
  @(negedge ctl.clock);
  ctl.query_0 = '0;
  ctl.query_1 = '0;
endtask
task wait_ready(int id, int error = 0);
  while (!notified.exists(id)) @(negedge ctl.clock);
  verify_contract(status[id] == error, "wrong terminal status");
endtask
task retire_tag(int id, bit probe_release = 0);
  @(negedge ctl.clock);
  retire.bits = '0;
  retire.bits[`PF(RELEASE, ID)] = id;
  retire.valid = 1;
  if (probe_release) begin
    ctl.query_0 = query(id, 'h1010, 8, 0);
    ctl.query_1 = query(2, 'h4004, 8, 1);
  end
  do @(ctl.sample); while (!retire.sample.ready);
  @(negedge ctl.clock);
  retire.valid = 0;
  ctl.query_0  = '0;
  ctl.query_1  = '0;
endtask
task execute();
  ctl.reset = 1;
  ctl.hold_ready = 0;
  ctl.query_0 = '0;
  ctl.query_1 = '0;
  reserve.valid = 0;
  reserve.bits = '0;
  prepared.valid = 0;
  prepared.bits = '0;
  retire.valid = 0;
  retire.bits = '0;
  fork
    monitor();
  join_none
  repeat (3) @(negedge ctl.clock);
  ctl.reset = 0;
  reserve_tag(1);
  record(1, 'h1000, 'h8000, 4096, 0, 0);
  probes(query(1, 'h1010, 16, 0), query(1, 'h1010, 16, 1));
  @(negedge ctl.clock);
  ctl.hold_ready = 1;
  record(1, 'h2000, 'hc000, 4096, 0, 1);
  probes(query(1, 'h1010, 16, 0), query(1, 'h2010, 16, 0), 8);
  @(negedge ctl.clock);
  ctl.hold_ready = 0;
  wait_ready(1);
  probes(query(1, 'h1ffe, 2, 0), query(1, 'h2ffe, 2, 0));
  probes(query(1, 'h1fff, 2, 0), query(1, 'h3000, 4, 0));
  probes(query(1, 'h1000, 16, 1), query(1, 64'hffffffffffffffff, 2, 0));
  probes(query(1, 'h1000, 0, 0), query(99, 'h1000, 8, 0));
  reserve_tag(2);
  record(2, 'h4000, 'h9000, 64, 1, 0);
  record(2, 64'hffffffffffffffe0, (64'd1 << `PF_ADDRESS_BITS) - 32, 32, 1, 1);
  wait_ready(2);
  probes(query(1, 'h1f00, 16, 0), query(2, 'h4004, 16, 1));
  probes(query(2, 64'hfffffffffffffff8, 8, 1), query(2, 64'hfffffffffffffff8, 9, 1));
  retire_tag(1, 1);
  probes(query(1, 'h1010, 16, 0), query(2, 'h4004, 16, 1));
  reserve_tag(1);
  record(1, 'h6000, 'ha000, 128, 0, 1);
  wait_ready(1);
  probes(query(1, 'h1010, 16, 0), query(1, 'h6040, 64, 0));
  reserve_tag(3);
  record(3, 'h7000, 'hb000, 64, 0, 0);
  record(3, 'h8000, 0, 0, 0, 1, 3);
  wait_ready(3, 3);
  probes(query(3, 'h7004, 8, 0), query(3, 'h8000, 8, 0));
  reserve_tag(4);
  fork
    reserve_tag(5);
    begin
      repeat (6) begin
        @(negedge ctl.clock);
        verify_contract(!reserve.ready, "full table accepted fifth reserve");
      end
      record(4, 'h9000, 0, 0, 0, 1, 4);
      wait_ready(4, 4);
      retire_tag(4);
    end
  join
  for (int i = 0; i < 8; i++) begin
    record(5, 'h10000 + i * 4096, 'h30000 + (7 - i) * 4096, 32, 0, i == 7);
  end
  wait_ready(5);
  probes(query(5, 'h17018, 8, 0), query(5, 'h10008, 8, 0));
  retire_tag(1);
  retire_tag(2);
  retire_tag(3);
  retire_tag(5);
  for (int code = 1; code <= 6; code++) begin
    reserve_tag(20 + code);
    record(20 + code, 'h9000 + code * 16, 0, 0, code % 2, 1, code);
    wait_ready(20 + code, code);
    retire_tag(20 + code);
  end
  // Retain an explicitly authorized same-page union and a discontiguous next PA page.
  reserve_tag(10);
  record(10, 'h74003e00, 'h800e00, 512, 0, 0);
  record(10, 'h74004000, 'h802000, 16, 0, 1);
  wait_ready(10);
  probes(query(10, 'h74003e10, 16, 0), query(10, 'h74004000, 16, 0));
  probes(query(10, 'h74003df0, 16, 0), query(10, 'h74004010, 8, 0));
  probes(query(10, 'h74003ff8, 16, 0), query(10, 'h74003e48, 16, 1));
  retire_tag(10);
  reserve_tag(9);
  record(9, 'h9000, 'h19000, 64, 0, 1);
  // Reset cancels a terminal notification still under backpressure.
  @(negedge ctl.clock);
  ctl.hold_ready = 1;
  repeat (2) @(negedge ctl.clock);
  cancelled += pmap_ref_pending(model);
  ctl.reset = 1;
  repeat (3) @(negedge ctl.clock);
  ctl.reset = 0;
  ctl.hold_ready = 0;
  probes(query(9, 'h9000, 8, 0), query(1, 'h6000, 8, 0));
  repeat (3) @(negedge ctl.clock);
  verify_contract(pmap_ref_pending(model) == 0, "mapping remained after retirement/reset");
  verify_contract(hits > 0 && misses > 0 && dual > 0 && stalls >= 4 && cancelled == 1,
                  "required checks did not execute");
  `uvm_info(
      "PREPARED_MAP",
      $sformatf(
          "Checked %0d queries, %0d hits, %0d misses; reserve=%0d ready=%0d retire=%0d stalls=%0d dual=%0d reset_cancel=%0d",
          queries, hits, misses, allocations, notifications, retired, stalls, dual, cancelled),
      UVM_LOW)
endtask
