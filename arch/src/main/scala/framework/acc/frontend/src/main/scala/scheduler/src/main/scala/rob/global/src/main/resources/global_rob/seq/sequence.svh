class scenario_sequence extends uvm_sequence;
  `uvm_object_utils(scenario_sequence)
  env test_env;
  virtual global_if vif;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task send(int serial, int kind, int write_bank, int read0, int read1);
    command_sequence seq = command_sequence::type_id::create("command");
    command_t packet = '0;
    longint unsigned rs1, rs2;
    int funct;
    bit wr, rd0, rd1, update, config_op, movement;
    operation_t operation;
    wr = kind == 1 || kind == 3 || kind == 4 || kind == 5 || kind == 6 || kind == 7;
    rd0 = kind == 2 || kind == 3 || kind == 4 || kind == 5 || kind == 7;
    rd1 = kind == 4 || kind == 5 || kind == 7;
    update = kind == 5;
    config_op = kind == 6 || kind == 7;
    movement = kind == 8;
    if (kind == 6) read0 = write_bank;
    if (kind == 7) read1 = write_bank;
    case (kind)
      0: funct = 5;
      1: funct = 'h24;
      2: funct = 'h14;
      3: funct = 'h34;
      4, 5: funct = 'h44;
      6, 7: funct = 32;
      8: funct = 13;
      default: `uvm_fatal("KIND", "Unknown global ROB scenario command")
    endcase
    rs1=(64'(serial%17+1)<<30)|(64'(write_bank)<<20)|(64'(read1)<<10)|read0|(update?64'h8000000000000000:0);
    rs2 = config_op ? (kind == 7 ? 64'h1002 : 64'h2) : (64'h123456789abcdef0 ^ serial);
    for (int bit_index = 0; bit_index < `GROB_IN_CONTEXT_BITS; bit_index++)
      packet[`GROB_IN_CONTEXT_OFFSET-`GROB_IN_ALLOC_OFFSET+bit_index]=(serial>>(bit_index%16))^((bit_index+serial)%3==0);
    operation = '0;
    `OP(operation, RUNTIME) = serial % 5 != 0;
    `OP(operation, DOMAIN_ID) = config_op || movement ? 1 : 3;
    `OP(operation, ITER) = serial % 17 + 1;
    `OP(operation, CMD_RAW_INST) = (funct << 25) | 'h7b;
    `OP(operation, CMD_PC) = 64'hdead000010000000 + 64'(serial) * 4;
    `OP(operation, CMD_RS1DATA) = rs1;
    `OP(operation, CMD_RS2DATA) = rs2;
    `OP(operation, BANKACCESS_WR_BANK_VALID) = wr;
    `OP(operation, BANKACCESS_WR_BANK_ID) = write_bank;
    `OP(operation, BANKACCESS_RD_BANK_0_VALID) = rd0;
    `OP(operation, BANKACCESS_RD_BANK_0_ID) = read0;
    `OP(operation, BANKACCESS_RD_BANK_1_VALID) = rd1;
    `OP(operation, BANKACCESS_RD_BANK_1_ID) = read1;
    `OP(operation, BANKACCESS_RD_WR_VALID) = update;
    `OP(operation, BANKACCESS_RD_WR_ID) = write_bank;
    `OP(operation, BANKACCESS_WR_UPDATE) = update;
    `OP(operation, ARCHITECTURALDESTINATION) = write_bank;
    packet[`GROB_IN_OPERATION_OFFSET-`GROB_IN_ALLOC_OFFSET+:`GROB_IN_OPERATION_BITS] = operation;
    seq.item = command_item::type_id::create("item");
    seq.item.payload = packet;
    seq.start(test_env.seqr);
  endtask
  task drain();
    do
      @(negedge vif.clock);
    while (test_env.reference.order.size() != 0 || !
    `GO(test_env.reference.latest.outputs, EMPTY)
    );
    repeat (3) @(negedge vif.clock);
  endtask
  task body();
    int before_delivered;
    test_env.reference.hold_complete = 1;
    test_env.reference.hold_issue = 1;
    test_env.reference.reverse_complete = 1;
    test_env.reset.pulse(4);
    for (int i = 0; i < `GROB_ENTRIES; i++) send(i + 1, 1, i % `GROB_VBANKS, 0, 0);
    repeat (5) @(negedge vif.clock);
    evidence::check(`GO(test_env.reference.latest.outputs, FULL),
                    "Global ROB did not reach full occupancy");
    fork
      send(`GROB_ENTRIES + 1, 1, 0, 0, 0);
      begin
        repeat (5) @(negedge vif.clock);
        evidence::check(!`GO(test_env.reference.latest.outputs, ALLOC_READY),
                        "Global ROB admitted work while full");
        test_env.reference.hold_issue = 0;
        test_env.reference.hold_complete = 0;
      end
    join
    drain();
    before_delivered = test_env.reference.delivered;
    test_env.reference.hold_mapping = 1;
    send(6, 1, 0, 0, 0);
    send(7, 2, 0, 1 % `GROB_VBANKS, 0);
    repeat (10) @(negedge vif.clock);
    evidence::check(test_env.reference.delivered == before_delivered &&
                    `GO(test_env.reference.latest.outputs, MAPPING_VALID),
                    "Global ROB bypassed a blocked oldest mapping request");
    test_env.reference.hold_mapping = 0;
    drain();
    for (int n = 10; n < 510; n++)
      send(n, n % 9, (n / 3) % `GROB_VBANKS, n % `GROB_VBANKS, (n + 1) % `GROB_VBANKS);
    drain();
    // A partially selected issue pipeline and all its queued work cancel on reset.
    test_env.reference.hold_issue = 1;
    test_env.reference.hold_complete = 1;
    for (int n = 0; n < `GROB_ENTRIES; n++) send(600 + n, 1, n % `GROB_VBANKS, 0, 0);
    repeat (6) @(negedge vif.clock);
    test_env.reset.pulse(4);
    test_env.reference.hold_issue = 0;
    test_env.reference.hold_complete = 0;
    for (int n = 0; n < 32; n++)
      send(700 + n, n % 6, n % `GROB_VBANKS, (n + 1) % `GROB_VBANKS, (n + 2) % `GROB_VBANKS);
    drain();
    evidence::check(evidence::cancelled == `GROB_ENTRIES,
                    "Global ROB reset cancellation count mismatch");
    `uvm_info(
        "GLOBAL_ROB",
        $sformatf(
            "allocated=%0d selected=%0d delivered=%0d completed=%0d retired=%0d reset_cancelled=%0d reordered=%0d",
            test_env.reference.allocated, test_env.reference.selected,
            test_env.reference.delivered, test_env.reference.completed, test_env.reference.retired,
            test_env.reference.cancelled_live, test_env.reference.reordered), UVM_LOW)
  endtask
endclass
