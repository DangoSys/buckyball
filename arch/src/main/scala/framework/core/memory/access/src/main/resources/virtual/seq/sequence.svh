class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  model reference;
  logic [`VA_INPUT_BITS-1:0] pins;
  logic [`VA_OUTPUT_BITS-1:0] observed;
  int tag;
  function new(string name = "scenario");
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
  task transact(longint unsigned address, int size, bit writing = 0, bit signed_load = 0,
                int atomic_op = 0, bit executing = 0, int privilege = 3, int mode = 0, bit sum = 0,
                bit mxr = 0, int cancel_after = 0);
    pins[`VA_IN_REQUEST_BITS_VADDR_OFFSET+:`VA_IN_REQUEST_BITS_VADDR_WIDTH] = address;
    pins[`VA_IN_REQUEST_BITS_SIZE_OFFSET+:`VA_IN_REQUEST_BITS_SIZE_WIDTH] = size;
    pins[`VA_IN_REQUEST_BITS_TAG_OFFSET+:`VA_IN_REQUEST_BITS_TAG_WIDTH] = tag++;
    pins[`VA_IN_REQUEST_BITS_WRITE_OFFSET+:`VA_IN_REQUEST_BITS_WRITE_WIDTH] = writing;
    pins[`VA_IN_REQUEST_BITS_SIGNED_OFFSET+:`VA_IN_REQUEST_BITS_SIGNED_WIDTH] = signed_load;
    pins[`VA_IN_REQUEST_BITS_ATOMIC_OFFSET+:`VA_IN_REQUEST_BITS_ATOMIC_WIDTH] = atomic_op;
    pins[`VA_IN_REQUEST_BITS_EXECUTE_OFFSET+:`VA_IN_REQUEST_BITS_EXECUTE_WIDTH] = executing;
    pins[`VA_IN_REQUEST_BITS_DATA_OFFSET+:`VA_IN_REQUEST_BITS_DATA_WIDTH] = 64'hf1e2d3c4b5a69788;
    pins[`VA_IN_REQUEST_BITS_PRIVILEGE_OFFSET+:`VA_IN_REQUEST_BITS_PRIVILEGE_WIDTH] = privilege;
    pins[`VA_IN_REQUEST_BITS_SATPMODE_OFFSET+:`VA_IN_REQUEST_BITS_SATPMODE_WIDTH] = mode;
    pins[`VA_IN_REQUEST_BITS_ROOTPPN_OFFSET+:`VA_IN_REQUEST_BITS_ROOTPPN_WIDTH] = 16;
    pins[`VA_IN_REQUEST_BITS_SUM_OFFSET+:`VA_IN_REQUEST_BITS_SUM_WIDTH] = sum;
    pins[`VA_IN_REQUEST_BITS_MXR_OFFSET+:`VA_IN_REQUEST_BITS_MXR_WIDTH] = mxr;
    pins[`VA_IN_FLUSHTRANSLATION_OFFSET+:`VA_IN_FLUSHTRANSLATION_WIDTH] = 1;
    tick();
    pins[`VA_IN_FLUSHTRANSLATION_OFFSET+:`VA_IN_FLUSHTRANSLATION_WIDTH] = 0;
    pins[`VA_IN_REQUEST_VALID_OFFSET+:`VA_IN_REQUEST_VALID_WIDTH] = 1;
    pins[`VA_IN_RESPONSE_READY_OFFSET+:`VA_IN_RESPONSE_READY_WIDTH] = 0;
    do tick(); while (!observed[`VA_OUT_REQUEST_READY_OFFSET]);
    pins[`VA_IN_REQUEST_VALID_OFFSET+:`VA_IN_REQUEST_VALID_WIDTH] = 0;
    if (cancel_after > 0) begin
      repeat (cancel_after) tick();
      repeat (2) tick(1);
      tick();
      return;
    end
    do tick(); while (!observed[`VA_OUT_RESPONSE_VALID_OFFSET]);
    repeat (4) tick();
    pins[`VA_IN_RESPONSE_READY_OFFSET+:`VA_IN_RESPONSE_READY_WIDTH] = 1;
    tick();
    tick();
  endtask
  task body();
    pins = '0;
    pins[`VA_IN_ACTIVE_OFFSET+:`VA_IN_ACTIVE_WIDTH] = 1;
    repeat (3) tick(1);
    tick();
    for (int size = 0; size < 4; size++)
      for (int offset = 0; offset < 16; offset++)
        for (int direction = 0; direction < 2; direction++) begin
          transact('h2000 + offset, size, direction, 0);
          transact('h200000 + offset, size, direction, 1);
        end
    for (int atomic_op = 1; atomic_op <= 11; atomic_op++) begin
      transact('h2000, 2, 0, 0, atomic_op);
      transact('h2008, 3, 0, 0, atomic_op);
    end
    transact('h2000, 1, 0, 0, 0, 1);
    transact('h200000, 1, 0, 0, 0, 1);
    transact('h300000, 3);
    transact('h300000, 3, 1);
    transact('h500000, 3);
    for (int kind = 0; kind < 2; kind++) begin
      reference.image.deny_data = kind == 0;
      reference.image.backend_error = kind == 1;
      transact('h2000, 3);
      transact('h200000, 3);
      reference.image.deny_data = 0;
      reference.image.backend_error = 0;
    end
    reference.set_pte('h10000, ('h11 << 10) | 1);
    reference.set_pte('h11000, ('h12 << 10) | 1);
    reference.set_pte('h12008, ('h2 << 10) | 'hcf);
    for (int privilege = 0; privilege < 4; privilege++)
      if (privilege != 2) begin
        transact('h1000, 3, 0, 0, 0, 0, privilege, 8);
        transact('h1000, 3, 1, 0, 0, 0, privilege, 8);
      end
    transact('h1002, 1, 0, 0, 0, 1, 1, 8);
    transact(64'h4000000000, 3, 0, 0, 0, 0, 1, 8);
    reference.set_pte('h12008, ('h2 << 10) | 'hdf);
    transact('h1000, 3, 0, 0, 0, 0, 0, 8);
    transact('h1000, 3, 0, 0, 0, 0, 1, 8, 1);
    reference.set_pte('h12008, ('h2 << 10) | 'hc9);
    transact('h1000, 3, 0, 0, 0, 0, 1, 8, 0, 1);
    transact('h1000, 3, 0, 0, 0, 0, 1, 8);
    reference.set_pte('h12008, 0);
    transact('h1000, 3, 0, 0, 0, 0, 1, 8);
    reference.set_pte('h12008, ('h2 << 10) | 'hcf, 1);
    transact('h1000, 3, 0, 0, 0, 0, 1, 8);
    reference.set_pte('h12008, ('h2 << 10) | 'hcf);
    reference.image.deny_pte = 1;
    transact('h1000, 3, 0, 0, 0, 0, 1, 8);
    reference.image.deny_pte = 0;
    transact('h2000, 3, 0, 0, 0, 0, 3, 0, 0, 0, 2);
    transact('h2000, 3, 0, 0, 0, 0, 3, 0, 0, 0, 7);
    transact('h2000, 3);
    repeat (8) tick();
  endtask
endclass
