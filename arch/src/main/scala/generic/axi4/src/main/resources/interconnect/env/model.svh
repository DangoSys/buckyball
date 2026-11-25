class routing_model extends uvm_subscriber #(snapshot);
  `uvm_component_utils(routing_model)
  typedef struct {
    address addr;
    int master;
    int beats;
  } transaction;
  transaction reads[$], writes[$], owners[$];
  transaction read_slots[int], write_slots[int];
  transaction read_order[longint][$], write_order[longint][$];
  int live_reads = 0, live_writes = 0;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  covergroup behavior with function sample (int length, bit read, int owner, int response_code);
    option.per_instance = 1;
    burst_length: coverpoint length {
      bins one = {1}; bins short_burst = {[2 : 16]}; bins maximum = {256};
    }
    direction: coverpoint read;
    master: coverpoint owner {bins each_master[] = {[0 : M - 1]};}
    response: coverpoint response_code {bins codes[] = {0, 2, 3};}
  endgroup
  covergroup load with function sample (int active, int live, bit stalled);
    option.per_instance = 1;
    concurrency: coverpoint active {bins single = {1}; bins competing = {[2 : M]};}
    occupancy: coverpoint live {
      bins full = {2 * `AXI_SLOTS}; bins active = {[1 : 2 * `AXI_SLOTS - 1]};
    }
    backpressure: coverpoint stalled;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    behavior = new();
    load = new();
  endfunction
  function void compare_value(logic [D+128:0] expected, logic [D+128:0] actual, int channel,
                              int owner);
    result_item e = result_item::type_id::create("expected");
    result_item a = result_item::type_id::create("actual");
    e.payload = expected;
    a.payload = actual;
    e.channel = channel;
    a.channel = channel;
    e.owner   = owner;
    a.owner   = owner;
    expected_ap.write(e);
    actual_ap.write(a);
  endfunction
  function void write(snapshot s);
    transaction t;
    address original;
    int active, routed, slot;
    longint key;
    transaction retired;
    read_data response_data;
    response response_status;
    if (s.reset) begin
      evidence::check(live_reads == 0 && live_writes == 0, "Reset with unretired AXI transactions");
      return;
    end
    evidence::check(s.outstanding == live_reads + live_writes, "AXI outstanding count mismatch");
    active = 0;
    for (int m = 0; m < M; m++) begin
      if (s.ports[m].ar_valid || s.ports[m].aw_valid) active++;
      if (s.ports[m].ar_valid && s.ports[m].ar_ready) begin
        t.addr = s.ports[m].ar;
        t.master = m;
        t.beats = 0;
        key = (longint'(m) << I) | t.addr.id;
        if (!read_order.exists(key)) read_order[key] = {};
        read_order[key].push_back(t);
        reads.push_back(t);
        live_reads++;
      end
      if (s.ports[m].aw_valid && s.ports[m].aw_ready) begin
        t.addr = s.ports[m].aw;
        t.master = m;
        t.beats = 0;
        key = (longint'(m) << I) | t.addr.id;
        if (!write_order.exists(key)) write_order[key] = {};
        write_order[key].push_back(t);
        writes.push_back(t);
        owners.push_back(t);
        live_writes++;
      end
    end
    load.sample(active, s.outstanding,
                (s.ports[M].ar_valid && !s.ports[M].ar_ready) ||
      (s.ports[M].aw_valid && !s.ports[M].aw_ready) ||
      (s.ports[M].r_valid && !s.ports[M].r_ready) ||
      (s.ports[M].b_valid && !s.ports[M].b_ready));
    if (s.ports[M].ar_valid && s.ports[M].ar_ready) begin
      if (reads.size() == 0) `uvm_fatal("AR", "Unrequested downstream read")
      t = reads.pop_front();
      original = s.ports[M].ar;
      slot = original.id;
      original.id = t.addr.id;
      evidence::check(slot < `AXI_SLOTS && !read_slots.exists(slot),
                      "Read slot reused before retirement");
      compare_value(t.addr, original, 0, t.master);
      read_slots[slot] = t;
    end
    if (s.ports[M].aw_valid && s.ports[M].aw_ready) begin
      if (writes.size() == 0) `uvm_fatal("AW", "Unrequested downstream write")
      t = writes.pop_front();
      original = s.ports[M].aw;
      slot = original.id;
      original.id = t.addr.id;
      evidence::check(slot < `AXI_SLOTS && !write_slots.exists(slot),
                      "Write slot reused before retirement");
      compare_value(t.addr, original, 1, t.master);
      write_slots[slot] = t;
    end
    routed = 0;
    for (int m = 0; m < M; m++)
    if (s.ports[m].w_valid && s.ports[m].w_ready) begin
      if (owners.size() == 0) `uvm_fatal("W", "Write data without an accepted address")
      evidence::check(owners[0].master == m && s.ports[M].w_valid && s.ports[M].w_ready,
                      "Write data routed to the wrong owner");
      compare_value(s.ports[m].w, s.ports[M].w, 2, m);
      evidence::check(s.ports[m].w.last == (owners[0].beats == owners[0].addr.len),
                      "Write burst length mismatch");
      owners[0].beats++;
      if (s.ports[m].w.last) t = owners.pop_front();
      routed++;
    end
    evidence::check(routed == int'(s.ports[M].w_valid && s.ports[M].w_ready),
                    "Lost or duplicated W beat");
    routed = 0;
    for (int m = 0; m < M; m++)
    if (s.ports[m].r_valid && s.ports[m].r_ready) begin
      slot = s.ports[M].r.id;
      if (!read_slots.exists(slot)) `uvm_fatal("R", "Response for an unissued read")
      t   = read_slots[slot];
      key = (longint'(t.master) << I) | t.addr.id;
      evidence::check(read_order[key][0].addr == t.addr,
                      "Read responses reordered within an original ID");
      evidence::check(t.master == m && s.ports[M].r_valid && s.ports[M].r_ready,
                      "Read response routed to the wrong master");
      response_data = s.ports[M].r;
      response_data.id = t.addr.id;
      compare_value(response_data, s.ports[m].r, 3, m);
      evidence::check(response_data.last == (t.beats == t.addr.len), "Read burst length mismatch");
      behavior.sample(t.addr.len + 1, 1, m, response_data.resp);
      read_slots[slot].beats++;
      if (response_data.last) begin
        retired = read_order[key].pop_front();
        if (read_order[key].size() == 0) read_order.delete(key);
        read_slots.delete(slot);
        live_reads--;
      end
      routed++;
    end
    evidence::check(routed == int'(s.ports[M].r_valid && s.ports[M].r_ready),
                    "Lost or duplicated R beat");
    routed = 0;
    for (int m = 0; m < M; m++)
    if (s.ports[m].b_valid && s.ports[m].b_ready) begin
      slot = s.ports[M].b.id;
      if (!write_slots.exists(slot)) `uvm_fatal("B", "Response for an unissued write")
      t   = write_slots[slot];
      key = (longint'(t.master) << I) | t.addr.id;
      evidence::check(write_order[key][0].addr == t.addr,
                      "Write responses reordered within an original ID");
      evidence::check(t.master == m && s.ports[M].b_valid && s.ports[M].b_ready,
                      "Write response routed to the wrong master");
      response_status = s.ports[M].b;
      response_status.id = t.addr.id;
      compare_value(response_status, s.ports[m].b, 4, m);
      behavior.sample(t.addr.len + 1, 0, m, response_status.resp);
      retired = write_order[key].pop_front();
      if (write_order[key].size() == 0) write_order.delete(key);
      write_slots.delete(slot);
      live_writes--;
      routed++;
    end
    evidence::check(routed == int'(s.ports[M].b_valid && s.ports[M].b_ready),
                    "Lost or duplicated B response");
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    evidence::check(live_reads == 0 && live_writes == 0 && owners.size() == 0,
                    "AXI transactions remain outstanding");
  endfunction
endclass
