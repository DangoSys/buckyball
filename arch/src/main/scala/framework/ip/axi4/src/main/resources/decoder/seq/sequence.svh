class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  port_signals pins[3] = '{default: '0}, observed[3];
  const logic [`AXI_ADDRESS_BITS-1:0] bases[`DECODER_REGIONS] = `DECODER_BASES;
  const logic [`AXI_ADDRESS_BITS-1:0] sizes[`DECODER_REGIONS] = `DECODER_SIZES;
  int serial;
  function new(string name = "scenario");
    super.new(name);
  endfunction
  task tick(bit reset = 0);
    cycle_item t = cycle_item::type_id::create("cycle");
    start_item(t);
    t.ports = pins;
    t.reset = reset;
    finish_item(t);
    observed = t.ports;
  endtask
  task burst(logic [`AXI_ADDRESS_BITS-1:0] read_address,
             logic [`AXI_ADDRESS_BITS-1:0] write_address, int beats, bit cancel = 0);
    bit read_sent, write_sent;
    int rtarget, wtarget;
    pins = '{default: '0};
    pins[0].ar = '1;
    pins[0].ar.addr = read_address;
    pins[0].ar.len = beats - 1;
    pins[0].ar.id = I'(serial);
    pins[0].ar.burst = 1;
    pins[0].aw = '0;
    pins[0].aw.addr = write_address;
    pins[0].aw.len = beats - 1;
    pins[0].aw.id = I'(~serial);
    pins[0].aw.burst = 1;
    pins[0].w = '1;
    pins[0].w.last = beats == 1;
    pins[0].w_valid = 1;
    repeat (3) tick();
    pins[0].ar_valid = 1;
    pins[0].aw_valid = 1;
    for (int cycle = 0; !read_sent || !write_sent; cycle++) begin
      pins[1].ar_ready = cycle % 3 != 0;
      pins[2].ar_ready = cycle % 3 == 0;
      pins[1].aw_ready = cycle % 4 == 0;
      pins[2].aw_ready = cycle % 4 != 0;
      tick();
      if (!read_sent && observed[0].ar_ready) begin
        read_sent = 1;
        pins[0].ar_valid = 0;
        rtarget = observed[1].ar_valid ? 1 : 2;
      end
      if (!write_sent && observed[0].aw_ready) begin
        write_sent = 1;
        pins[0].aw_valid = 0;
        wtarget = observed[1].aw_valid ? 1 : 2;
      end
    end
    // Offered addresses keep changing after acceptance; response owners stay captured.
    pins[0].ar.addr = ~read_address;
    pins[0].aw.addr = ~write_address;
    if (cancel) begin
      repeat (2) tick(1);
      pins = '{default: '0};
      tick();
      return;
    end
    for (int beat = 0; beat < beats; beat++) begin
      pins[rtarget].r_valid = 1;
      pins[rtarget].r.id = pins[0].ar.id;
      pins[rtarget].r.data = D'(64'h123456789abcdef0) ^ beat ^ serial;
      pins[rtarget].r.resp = 2'(serial);
      pins[rtarget].r.last = beat == beats - 1;
      pins[0].w.data = ~pins[rtarget].r.data;
      pins[0].w.strb = (beat % 2) ? '1 : 'h55;
      pins[0].w.last = beat == beats - 1;
      pins[0].w_valid = 1;
      // Idle-branch traffic must not leak onto the selected response path.
      pins[3-rtarget].r_valid = 1;
      pins[3-rtarget].r = '1;
      pins[0].r_ready = 0;
      pins[wtarget].w_ready = 0;
      repeat (3) tick();
      pins[0].r_ready = 1;
      pins[wtarget].w_ready = 1;
      tick();
      pins[rtarget].r_valid = 0;
      pins[3-rtarget].r_valid = 0;
      pins[0].w_valid = 0;
    end
    pins[wtarget].b_valid = 1;
    pins[wtarget].b.id = pins[0].aw.id;
    pins[wtarget].b.resp = 2'(~serial);
    pins[3-wtarget].b_valid = 1;
    pins[3-wtarget].b = '1;
    pins[0].b_ready = 0;
    repeat (3) tick();
    pins[0].b_ready = 1;
    tick();
    pins = '{default: '0};
    tick();
    serial++;
  endtask
  task body();
    repeat (3) tick(1);
    tick();
    foreach (bases[i]) begin
      burst(bases[i] - 1, bases[i], 1);
      burst(bases[i], bases[i] + sizes[i] - 1, 2);
      burst(bases[i] + sizes[i] - 1, bases[i] + sizes[i], 3);
      burst(bases[i] + sizes[i], bases[i] - 1, 4);
    end
    for (int i = 0; i < 32; i++)
      burst(i % 2 ? bases[i%`DECODER_REGIONS] : 32'h80000000 + i * 64,
            i % 3 ? 32'h90000000 + i * 64 : bases[i%`DECODER_REGIONS], 1 + i % 8);
    burst(bases[0], 32'h80000000, 4, 1);
    burst(32'h80000000, bases[0], 4, 1);
    burst(bases[1], bases[2], 4);
  endtask
endclass
