class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  logic [`BOOT_INPUT_BITS-1:0] pins = '0;
  logic [`BOOT_OUTPUT_BITS-1:0] observed;
  bit reset = 1;
  int pending_work = 0, cycle = 0;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task tick(int cycles = 1);
    cycle_item req;
    repeat (cycles) begin
      `BI(pins, SCHEDULERIDLE) = pending_work == 0;
      req = cycle_item::type_id::create("cycle");
      start_item(req);
      req.inputs = pins;
      req.reset  = reset;
      finish_item(req);
      observed = req.outputs;
      if (pending_work > 0) pending_work--;
      if (!reset && `BO(observed, CMD_VALID) && `BI(pins, CMD_READY)) pending_work = 3;
      cycle++;
    end
  endtask
  task restart();
    pins = '0;
    reset = 1;
    pending_work = 0;
    tick(2);
    reset = 0;
    tick();
  endtask
  task send(logic [`BOOT_PACKET_BITS-1:0] payload);
    pins[`BOOT_IN_PACKET_OFFSET+:`BOOT_PACKET_BITS] = payload;
    `BI(pins, COMMAND_VALID) = 1;
    do begin
      `BI(pins, CMD_READY) = cycle % 3 == 0;
      tick();
    end while (!
    `BO(observed, COMMAND_READY)
    );
    `BI(pins, COMMAND_VALID) = 0;
  endtask
  task drain();
    `BI(pins, CMD_READY) = 1;
    do tick(); while (!`BO(observed, IDLE));
  endtask
  task body();
    logic [`BOOT_PACKET_BITS-1:0] packet;
    restart();
    // A runtime producer may hold VALID while the ROM is still initializing balls.
    packet = '1;
    packet[`BOOT_IN_COMMAND_BITS_INSTRUCTION_RAW_INST_OFFSET-`BOOT_IN_PACKET_OFFSET+:32] = 32'h4a00007b;
    send(packet);
    drain();
    for (int bit_index = 0; bit_index < `BOOT_PACKET_BITS; bit_index++) begin
      packet = (`BOOT_PACKET_BITS'(1) << bit_index) ^ ((bit_index % 2) ? '1 : '0);
      packet[`BOOT_IN_COMMAND_BITS_INSTRUCTION_RAW_INST_OFFSET-`BOOT_IN_PACKET_OFFSET+:32] = (32'(bit_index%128)<<25) | 'h7b;
      send(packet);
    end
    drain();
    // Restart after only one initialization command; the remaining expected commands are cancelled.
    restart();
    `BI(pins, CMD_READY) = 1;
    do tick(); while (!`BO(observed, CMD_VALID));
    restart();
    drain();
  endtask
endclass
