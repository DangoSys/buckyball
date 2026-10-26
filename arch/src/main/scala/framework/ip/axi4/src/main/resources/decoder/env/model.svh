class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(result_item) expected, actual;
  const logic [`AXI_ADDRESS_BITS-1:0] bases[`DECODER_REGIONS] = `DECODER_BASES;
  const logic [`AXI_ADDRESS_BITS-1:0] sizes[`DECODER_REGIONS] = `DECODER_SIZES;
  bit read_busy, write_busy, data_pending;
  int read_target, write_target;
  covergroup behavior with function sample (
      int read_target, int write_target, bit concurrent, bit held, bit leading_data, bit reset_busy
  );
    option.per_instance = 1;
    read_routes: coverpoint read_target {bins all[] = {1, 2};}
    write_routes: coverpoint write_target {bins all[] = {1, 2};}
    overlap: coverpoint concurrent;
    backpressure: coverpoint held;
    data_before_address: coverpoint leading_data;
    reset_cancellation: coverpoint reset_busy;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual   = new("actual", this);
    behavior = new();
  endfunction
  function int branch(logic [`AXI_ADDRESS_BITS-1:0] address);
    foreach (bases[i]) if (address >= bases[i] && address - bases[i] < sizes[i]) return 1;
    return 2;
  endfunction
  function logic [3*$bits(port_signals)-1:0] outputs(port_signals ports[3]);
    port_signals result[3] = '{default: '0};
    result[0].aw_ready = ports[0].aw_ready;
    result[0].ar_ready = ports[0].ar_ready;
    result[0].w_ready  = ports[0].w_ready;
    result[0].r_valid  = ports[0].r_valid;
    result[0].b_valid  = ports[0].b_valid;
    if (ports[0].r_valid) result[0].r = ports[0].r;
    if (ports[0].b_valid) result[0].b = ports[0].b;
    for (int p = 1; p < 3; p++) begin
      result[p].aw_valid = ports[p].aw_valid;
      result[p].ar_valid = ports[p].ar_valid;
      result[p].w_valid  = ports[p].w_valid;
      result[p].r_ready  = ports[p].r_ready;
      result[p].b_ready  = ports[p].b_ready;
      if (ports[p].aw_valid) result[p].aw = ports[p].aw;
      if (ports[p].ar_valid) result[p].ar = ports[p].ar;
      if (ports[p].w_valid) result[p].w = ports[p].w;
    end
    return {result[2], result[1], result[0]};
  endfunction
  function void write(cycle_item t);
    port_signals want[3] = '{default: '0};
    int read_branch, write_branch;
    result_item predicted, received;
    if (t.reset) begin
      behavior.sample(1, 1, 0, 0, 0, read_busy || write_busy);
      read_busy = 0;
      write_busy = 0;
      data_pending = 0;
      return;
    end
    predicted = result_item::type_id::create("expected");
    received = result_item::type_id::create("actual");
    read_branch = branch(t.ports[0].ar.addr);
    write_branch = branch(t.ports[0].aw.addr);
    want[read_branch].ar_valid = t.ports[0].ar_valid && !read_busy;
    want[read_branch].ar = t.ports[0].ar;
    want[0].ar_ready = !read_busy && t.ports[read_branch].ar_ready;
    want[write_branch].aw_valid = t.ports[0].aw_valid && !write_busy;
    want[write_branch].aw = t.ports[0].aw;
    want[0].aw_ready = !write_busy && t.ports[write_branch].aw_ready;
    if (read_busy) begin
      want[0].r_valid = t.ports[read_target].r_valid;
      want[0].r = t.ports[read_target].r;
      want[read_target].r_ready = t.ports[0].r_ready;
    end
    if (write_busy) begin
      want[0].b_valid = t.ports[write_target].b_valid;
      want[0].b = t.ports[write_target].b;
      want[write_target].b_ready = t.ports[0].b_ready;
      if (data_pending) begin
        want[write_target].w_valid = t.ports[0].w_valid;
        want[write_target].w = t.ports[0].w;
        want[0].w_ready = t.ports[write_target].w_ready;
      end
    end
    predicted.value = outputs(want);
    received.value  = outputs(t.ports);
    expected.write(predicted);
    actual.write(received);
    behavior.sample(
        read_busy ? read_target : read_branch, write_busy ? write_target : write_branch,
        read_busy && write_busy,
        (t.ports[0].r_valid && !t.ports[0].r_ready) || (t.ports[0].w_valid && !t.ports[0].w_ready),
        t.ports[0].w_valid && !write_busy, 0);
    if (t.ports[0].ar_valid && t.ports[0].ar_ready) begin
      read_busy   = 1;
      read_target = read_branch;
    end
    if (t.ports[0].r_valid && t.ports[0].r_ready && t.ports[0].r.last) read_busy = 0;
    if (t.ports[0].aw_valid && t.ports[0].aw_ready) begin
      write_busy   = 1;
      data_pending = 1;
      write_target = write_branch;
    end
    if (t.ports[0].w_valid && t.ports[0].w_ready && t.ports[0].w.last) data_pending = 0;
    if (t.ports[0].b_valid && t.ports[0].b_ready) write_busy = 0;
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (read_busy || write_busy) `uvm_fatal("PENDING", "Decoder transaction did not drain")
  endfunction
endclass
