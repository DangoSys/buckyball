class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  logic [`SB_INPUT_BITS-1:0] pins = '0;
  bit reset = 1;
  int query_offsets[`SB_ENTRIES][6] = `SB_QUERY_OFFSETS;
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
  task queries(int mask, int bank, int alternate);
    for (int lane = 0; lane < `SB_ENTRIES; lane++)
      for (int port = 0; port < 3; port++) begin
        pins[query_offsets[lane][port*2]] = (mask >> port) & 1;
        pins[query_offsets[lane][port*2+1]+:`SB_BANKIDBITS] = (lane+port)%2 == 0 ? bank : alternate;
      end
  endtask
  task body();
    tick(2);
    reset = 0;
    for (int bank = 0; bank < `SB_BANKS; bank++) begin
      `BI(pins, ALLOC_VALID) = 1;
      `BI(pins, ALLOC_BITS_WR_BANK_VALID) = 1;
      `BI(pins, ALLOC_BITS_WR_BANK_ID) = bank;
      tick();
      `BI(pins, ALLOC_VALID) = 0;
      for (int mask = 0; mask < 8; mask++)
      for (int alternate = 0; alternate < `SB_BANKS; alternate++) begin
        queries(mask, bank, alternate);
        tick();
      end
      `BI(pins, COMPLETE_VALID) = 1;
      `BI(pins, COMPLETE_BITS_WR_BANK_VALID) = 1;
      `BI(pins, COMPLETE_BITS_WR_BANK_ID) = bank;
      tick();
      `BI(pins, COMPLETE_VALID) = 0;
      tick();
      `BI(pins, ISSUE_VALID) = 1;
      `BI(pins, ISSUE_BITS_WR_BANK_VALID) = 1;
      `BI(pins, ISSUE_BITS_WR_BANK_ID) = bank;
      tick();
      `BI(pins, ISSUE_VALID) = 0;
      queries(7, bank, (bank + 1) % `SB_BANKS);
      tick();
      // Retirement wins when the same alias is presented by issue and complete together.
      `BI(pins, ISSUE_VALID) = 1;
      `BI(pins, COMPLETE_VALID) = 1;
      tick();
      `BI(pins, ISSUE_VALID) = 0;
      `BI(pins, COMPLETE_VALID) = 0;
      tick();
    end
    // Read/read sharing never creates a write dependency.
    `BI(pins, ISSUE_BITS_WR_BANK_VALID) = 0;
    `BI(pins, COMPLETE_BITS_WR_BANK_VALID) = 0;
    for (int bank = 0; bank < `SB_BANKS; bank++) begin
      `BI(pins, ISSUE_VALID) = 1;
      `BI(pins, ISSUE_BITS_RD_BANK_0_VALID) = 1;
      `BI(pins, ISSUE_BITS_RD_BANK_0_ID) = bank;
      `BI(pins, ISSUE_BITS_RD_BANK_1_VALID) = 1;
      `BI(pins, ISSUE_BITS_RD_BANK_1_ID) = bank;
      `BI(pins, ISSUE_BITS_RD_WR_VALID) = 1;
      `BI(pins, ISSUE_BITS_RD_WR_ID) = bank;
      queries(7, bank, bank);
      tick(`SB_ENTRIES);
      `BI(pins, ISSUE_VALID) = 0;
      `BI(pins, COMPLETE_VALID) = 1;
      `BI(pins, COMPLETE_BITS_RD_BANK_0_VALID) = 1;
      `BI(pins, COMPLETE_BITS_RD_BANK_0_ID) = bank;
      `BI(pins, COMPLETE_BITS_RD_BANK_1_VALID) = 1;
      `BI(pins, COMPLETE_BITS_RD_BANK_1_ID) = bank;
      `BI(pins, COMPLETE_BITS_RD_WR_VALID) = 1;
      `BI(pins, COMPLETE_BITS_RD_WR_ID) = bank;
      tick(`SB_ENTRIES);
      `BI(pins, COMPLETE_VALID) = 0;
    end
    `BI(pins, ALLOC_VALID) = 1;
    `BI(pins, ALLOC_BITS_WR_BANK_VALID) = 1;
    tick();
    reset = 1;
    tick(2);
    pins  = '0;
    reset = 0;
    queries(7, 0, `SB_BANKS - 1);
    tick(2);
  endtask
endclass
