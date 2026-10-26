class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  const int offsets[`TSS_PORTS][13] = `TSS_PORT_OFFSETS;
  logic [`TSS_INPUT_BITS-1:0] pins;
  logic [`TSS_OUTPUT_BITS-1:0] observed;
  int serial;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task tick(bit reset = 0);
    cycle_item t = cycle_item::type_id::create("cycle");
    start_item(t);
    t.inputs = pins;
    t.reset  = reset;
    finish_item(t);
    observed = t.outputs;
  endtask
  task request(int p, logic [63:0] address, int size, bit writing, logic [`TSS_DATA_BITS-1:0] data,
               logic [`TSS_MASK_BITS-1:0] mask);
    pins[offsets[p][0]] = 1;
    pins[offsets[p][1]+:64] = address;
    pins[offsets[p][2]+:`TSS_SIZE_BITS] = size;
    pins[offsets[p][3]] = writing;
    pins[offsets[p][4]+:`TSS_DATA_BITS] = data;
    pins[offsets[p][5]+:`TSS_MASK_BITS] = mask;
  endtask
  task accept(int p);
    do tick(); while (!observed[offsets[p][6]]);
    pins[offsets[p][0]] = 0;
  endtask
  task access (int p, logic [63:0] address, int size, bit writing = 0,
               logic [`TSS_DATA_BITS-1:0] data = '0, logic [`TSS_MASK_BITS-1:0] mask = '0);
    request(p, address, size, writing, data, mask);
    accept(p);
    do tick(); while (!observed[offsets[p][7]]);
  endtask
  task together(bit writing, bit hold_responses);
    bit waiting[`TSS_PORTS];
    bit any_waiting;
    for (int p = 0; p < `TSS_PORTS; p++) begin
      request(p, `TSS_BASE + p * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS), writing,
              `TSS_DATA_BITS'(serial * 1234567 + p), '1);
      pins[offsets[p][10]] = !hold_responses;
      waiting[p] = 1;
    end
    serial++;
    do begin
      tick();
      any_waiting = 0;
      for (int p = 0; p < `TSS_PORTS; p++) begin
        if (waiting[p] && observed[offsets[p][6]]) begin
          waiting[p] = 0;
          pins[offsets[p][0]] = 0;
        end
        any_waiting |= waiting[p];
      end
    end while (any_waiting);
    if (hold_responses) begin
      bit all_valid;
      do begin
        tick();
        all_valid = 1;
        for (int p = 0; p < `TSS_PORTS; p++) all_valid &= observed[offsets[p][7]];
      end while (!all_valid);
    end else repeat (8) tick();
  endtask
  task body();
    pins = '0;
    for (int p = 0; p < `TSS_PORTS; p++) pins[offsets[p][10]] = 1;
    repeat (2) tick(1);
    tick();
    for (int p = 0; p < `TSS_CLIENTS; p++)
      request(p, `TSS_BASE + p * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS), 0, '0, '0);
    for (int row = 0; row < `TSS_BYTES / `TSS_MASK_BITS; row++) begin
      logic [`TSS_DATA_BITS-1:0] pattern;
      for (int b = 0; b < `TSS_MASK_BITS; b++) pattern[b*8+:8] = 8'(row * 17 + b);
      access (`TSS_CLIENTS, `TSS_BASE + row * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS), 1, pattern,
              '1);
    end
    pins[`TSS_IN_INUSE_OFFSET] = 1;
    begin
      bit waiting[`TSS_CLIENTS];
      bit any_waiting;
      foreach (waiting[p]) waiting[p] = 1;
      do begin
        tick();
        any_waiting = 0;
        foreach (waiting[p]) begin
          if (waiting[p] && observed[offsets[p][6]]) begin
            waiting[p] = 0;
            pins[offsets[p][0]] = 0;
          end
          any_waiting |= waiting[p];
        end
      end while (any_waiting);
      repeat (8) tick();
    end
    for (int size = 0; size <= $clog2(`TSS_MASK_BITS); size++)
      for (int address = 0; address < `TSS_BYTES; address += 1 << size)
        access ((address + size) % `TSS_PORTS, `TSS_BASE + address, size);
    for (int b = 0; b < `TSS_MASK_BITS; b++) begin
      access (b % `TSS_PORTS, `TSS_BASE, $clog2(`TSS_MASK_BITS), 1, ~`TSS_DATA_BITS'(b),
              `TSS_MASK_BITS'(1 << b));
      access ((b + 1) % `TSS_PORTS, `TSS_BASE, $clog2(`TSS_MASK_BITS));
    end
    access (0, `TSS_BASE, $clog2(`TSS_MASK_BITS), 1, '1, '0);
    access (0, `TSS_BASE, $clog2(`TSS_MASK_BITS));
    repeat (32) together(serial % 2, 0);
    pins[offsets[0][10]] = 0;
    request(0, `TSS_BASE, 0, 0, '0, '0);
    accept(0);
    do tick(); while (!observed[offsets[0][7]]);
    for (int n = 0; n < 10; n++) access (1 + n % (`TSS_PORTS - 1), `TSS_BASE + 64 + n, 0);
    pins[offsets[0][10]] = 1;
    tick();
    together(1, 1);
    pins[offsets[0][12]] = 1;
    tick();
    pins[offsets[0][12]] = 0;
    for (int p = 1; p < `TSS_PORTS; p++) pins[offsets[p][10]] = 1;
    repeat (8) tick();
    pins[offsets[0][10]] = 1;
    access (0, `TSS_BASE, $clog2(`TSS_MASK_BITS));
    for (int p = 0; p < `TSS_CLIENTS; p++) begin
      pins[offsets[p][10]] = 0;
      request(p, `TSS_BASE + 32 + p * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS), 1,
              `TSS_DATA_BITS'(64'hcafef00d + p), '1);
      accept(p);
      pins[offsets[p][12]] = 1;
      tick();
      pins[offsets[p][12]] = 0;
      do tick(); while (observed[offsets[p][11]]);
      pins[offsets[p][10]] = 1;
      access (p, `TSS_BASE + 32 + p * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS));
    end
    together(1, 1);
    repeat (2) tick(1);
    pins = '0;
    pins[`TSS_IN_INUSE_OFFSET] = 1;
    for (int p = 0; p < `TSS_PORTS; p++) pins[offsets[p][10]] = 1;
    tick();
    for (int p = 0; p < `TSS_PORTS; p++)
      access (p, `TSS_BASE + p * `TSS_MASK_BITS, $clog2(`TSS_MASK_BITS));
    pins[offsets[0][10]] = 0;
    request(0, `TSS_BASE, 0, 0, '0, '0);
    accept(0);
    pins[`TSS_IN_INUSE_OFFSET] = 0;
    pins[offsets[0][10]] = 1;
    repeat (6) tick();
    request(0, `TSS_BASE, 0, 0, '0, '0);
    access (`TSS_CLIENTS, `TSS_BASE + 128, 0);
    pins[offsets[0][0]] = 0;
    pins[`TSS_IN_INUSE_OFFSET] = 1;
    for (int p = 0; p < `TSS_PORTS; p++) begin
      access (p, `TSS_BASE - 1, 0);
      access (p, `TSS_BASE + `TSS_BYTES, 0);
      access (p, 64'hffffffffffffffff, 0);
      access (p, `TSS_BASE + 1, 1);
      access (p, `TSS_BASE, $clog2(`TSS_MASK_BITS) + 1);
      access (p, `TSS_BASE, 0, 1, '1, `TSS_MASK_BITS'(2));
    end
    repeat (12) tick();
  endtask
endclass
