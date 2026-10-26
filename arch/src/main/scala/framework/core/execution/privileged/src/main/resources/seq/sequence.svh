class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  logic [`PRIV_INPUT_BITS-1:0] pins = '0;
  bit reset = 1;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    repeat (cycles) begin
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.inputs = pins;
      req.reset  = reset;
      finish_item(req);
    end
  endtask
  task restart();
    pins  = '0;
    reset = 1;
    tick(2);
    reset = 0;
    `PIN(pins, HARTID) = 1;
    tick();
  endtask
  task csr(int address, bit write = 0, longint unsigned value = 0, int operation = 1,
           bit commit = 1);
    `PIN(pins, CSR_ADDRESS) = address;
    `PIN(pins, CSR_WRITE) = write;
    `PIN(pins, CSR_OPERAND) = value;
    `PIN(pins, CSR_OPERATION) = operation;
    `PIN(pins, CSR_COMMIT) = commit;
    tick();
    `PIN(pins, CSR_WRITE)  = 0;
    `PIN(pins, CSR_COMMIT) = 0;
  endtask
  task trap_event(longint unsigned pc, cause, value);
    `PIN(pins, TRAP_VALID) = 1;
    `PIN(pins, TRAP_BITS_PC) = pc;
    `PIN(pins, TRAP_BITS_CAUSE) = cause;
    `PIN(pins, TRAP_BITS_VALUE) = value;
    tick();
    `PIN(pins, TRAP_VALID) = 0;
  endtask
  task return_event(int mode);
    `PIN(pins, RETURNING_VALID) = 1;
    `PIN(pins, RETURNING_BITS)  = mode;
    tick();
    `PIN(pins, RETURNING_VALID) = 0;
  endtask
  task authorize(longint unsigned address, int size, int kind, int privilege);
    `PIN(pins, AUTHORIZATION_0_PADDR) = address;
    `PIN(pins, AUTHORIZATION_1_PADDR) = address;
    `PIN(pins, AUTHORIZATION_0_SIZE) = size;
    `PIN(pins, AUTHORIZATION_1_SIZE) = size;
    `PIN(pins, AUTHORIZATION_0_READ) = kind == 0;
    `PIN(pins, AUTHORIZATION_1_READ) = kind == 0;
    `PIN(pins, AUTHORIZATION_0_WRITE) = kind == 1;
    `PIN(pins, AUTHORIZATION_1_WRITE) = kind == 1;
    `PIN(pins, AUTHORIZATION_0_EXECUTE) = kind == 2;
    `PIN(pins, AUTHORIZATION_1_EXECUTE) = kind == 2;
    `PIN(pins, AUTHORIZATION_0_PRIVILEGE) = privilege;
    `PIN(pins, AUTHORIZATION_1_PRIVILEGE) = 3;
    `PIN(pins, AUTHORIZATION_0_ISPTE) = size % 2;
    `PIN(pins, AUTHORIZATION_1_ISPTE) = size % 2;
    tick();
  endtask
  task body();
    int registers[22] = '{
        'h105,
        'h140,
        'h141,
        'h142,
        'h143,
        'h180,
        'h300,
        'h302,
        'h303,
        'h304,
        'h305,
        'h306,
        'h310,
        'h320,
        'h340,
        'h341,
        'h342,
        'h343,
        'h344,
        'hb00,
        'hb02,
        'h106
    };
    int offsets[8] = '{-1, 0, 1, 3, 4, 63, 64, 128};
    restart();
    for (int address = 0; address < 4096; address++) csr(address);
    foreach (registers[index]) begin
      csr(registers[index], 1, 0);
      csr(registers[index], 1, '1);
      csr(registers[index], 1, 64'h0123456789abcdef, 1, 0);
      for (int bit_index = 0; bit_index < 64; bit_index++) begin
        csr(registers[index], 1, 64'(1) << bit_index, 2);
        csr(registers[index]);
        csr(registers[index], 1, 64'(1) << bit_index, 3);
      end
    end
    restart();
    csr('h300, 1, 'h2000);
    for (int rounding = 0; rounding < 8; rounding++) begin
      csr(2, 1, rounding);
      for (int flags = 0; flags < 32; flags++) begin
        csr(1, 1, 0);
        `PIN(pins, FLOATING_VALID) = 1;
        `PIN(pins, FLOATING_BITS)  = flags;
        tick();
        `PIN(pins, FLOATING_VALID) = 0;
        csr(3);
      end
    end
    // Every implemented PMP entry, addressing mode, access permission and lock state.
    for (int index = 0; index < 16; index++)
      for (int mode = 0; mode < 4; mode++)
        for (int permissions = 0; permissions < 8; permissions++)
          for (int locked = 0; locked < 2; locked++) begin
            longint unsigned base = 'h1000 + index * 'h100;
            longint unsigned encoded = (base >> 2) | (mode == 3 ? 7 : 0);
            int pmp_config = (locked << 7) | (mode << 3) | permissions;
            restart();
            if (mode == 1) begin
              if (index > 0) csr('h3b0 + index - 1, 1, base >> 2);
              encoded = (base + 64) >> 2;
            end
            csr('h3b0 + index, 1, encoded);
            csr('h3a0 + (index / 8) * 2, 1, 64'(pmp_config) << ((index % 8) * 8));
            for (int sample = 0; sample < 8; sample ++)
            for (int kind = 0; kind < 3; kind++)
            authorize(base + offsets[sample], sample, kind, sample % 2);
            csr('h3b0 + index, 1, '1);
            if (index > 0) csr('h3b0 + index - 1, 1, '1);
            csr('h3a0 + (index / 8) * 2, 1, 0);
          end
    // An empty or reversed TOR range must not claim an overlapping access.
    restart();
    csr('h3b0, 1, 'h410);
    csr('h3b1, 1, 'h400);
    csr('h3a0, 1, 64'h8f00);
    authorize('hfe0, 7, 0, 0);
    // CSR privilege checks cover the entire address space in S and U modes.
    for (int target = 0; target < 2; target++) begin
      restart();
      csr('h300, 1, target << 11);
      csr('h341, 1, 'h1000);
      return_event(3);
      for (int address = 0; address < 4096; address++) csr(address);
      trap_event('h12345, 64'h800000000000000b, 'hdeadbeef);
      csr('h341);
      csr('h342);
      csr('h343);
    end
    for (int target = 0; target < 2; target++)
      for (int vector_mode = 0; vector_mode < 2; vector_mode++) begin
        restart();
        csr('h305, 1, 'h1000 | vector_mode);
        csr('h105, 1, 'h2000 | vector_mode);
        csr('h302, 1, 'hb3ff);
        csr('h303, 1, 'h222);
        csr('h304, 1, 'haaa);
        csr('h300, 1, (target << 11) | 'h20);
        csr('h341, 1, 'h1000);
        return_event(3);
        trap_event('h12345, 8, 'hbad0);
        csr('h141);
        csr('h142);
        csr('h143);
        return_event(1);
        trap_event('h34567, 64'h800000000000000b, 'hbad1);
        csr('h341);
        return_event(3);
      end
    // Interrupt masking, delegation and priority across all external pending combinations.
    for (int target = 0; target < 4; target++) begin
      if (target == 2) continue;
      for (int enable = 0; enable < 4; enable++) begin
        restart();
        csr('h303, 1, 'h222);
        csr('h304, 1, 'haaa);
        csr('h344, 1, 'h222);
        csr('h300, 1, (target << 11) | ((enable & 1) << 3) | ((enable >> 1) << 1));
        if (target != 3) return_event(3);
        for (int pending = 0; pending < 16; pending++) begin
          `PIN(pins, INTERRUPTS_TIMER) = pending[0];
          `PIN(pins, INTERRUPTS_SOFTWARE) = pending[1];
          `PIN(pins, INTERRUPTS_EXTERNAL) = pending[2];
          `PIN(pins, INTERRUPTS_SUPERVISOREXTERNAL) = pending[3];
          tick();
        end
      end
    end
    restart();
    for (int inhibit = 0; inhibit < 8; inhibit++) begin
      csr('h320, 1, inhibit);
      for (int n = 0; n < 16; n++) begin
        `PIN(pins, RETIRED) = n % 2;
        csr('hb00);
        csr('hb02);
      end
    end
    `PIN(pins, RETIRED) = 0;
    csr('h320, 1, 0);
    tick(2);
  endtask
endclass
