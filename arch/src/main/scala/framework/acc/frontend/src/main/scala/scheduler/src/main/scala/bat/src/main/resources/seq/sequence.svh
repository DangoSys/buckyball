class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  localparam int V = `BAT_VIRTUALBANKS, A = 2 * V + `BAT_ROBENTRIES;
  logic [`BAT_INPUT_BITS-1:0] pins = '0;
  logic [`BAT_OUTPUT_BITS-1:0] observed;
  bit reset = 1;
  int stale[$];
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
      observed = req.outputs;
      if (!reset && `BI(pins, ALLOC_VALID)) begin
        if (`BO(observed, STALE_0_VALID)) stale.push_back(`BO(observed, STALE_0_BITS));
        if (`BO(observed, STALE_1_VALID)) stale.push_back(`BO(observed, STALE_1_BITS));
      end
    end
  endtask
  task restart();
    pins  = '0;
    reset = 1;
    stale.delete();
    tick(2);
    reset = 0;
    tick();
  endtask
  task allocate(int bank, bit rename, bit reset_read, bit update, int read0, read1, read_wr,
                int mask = 7);
    `BI(pins, ALLOC_VALID) = 1;
    `BI(pins, ALLOC_RENAME) = rename;
    `BI(pins, ALLOC_RESET_READ0) = reset_read;
    `BI(pins, ALLOC_RAW_WR_UPDATE) = update;
    `BI(pins, ALLOC_RAW_WR_BANK_VALID) = 1;
    `BI(pins, ALLOC_RAW_WR_BANK_ID) = bank;
    `BI(pins, ALLOC_RAW_RD_BANK_0_VALID) = mask[0];
    `BI(pins, ALLOC_RAW_RD_BANK_0_ID) = read0;
    `BI(pins, ALLOC_RAW_RD_BANK_1_VALID) = mask[1];
    `BI(pins, ALLOC_RAW_RD_BANK_1_ID) = read1;
    `BI(pins, ALLOC_RAW_RD_WR_VALID) = mask[2];
    `BI(pins, ALLOC_RAW_RD_WR_ID) = read_wr;
    tick();
    `BI(pins, ALLOC_VALID) = 0;
  endtask
  task reclaim_all();
    `BI(pins, ALLOC_VALID) = 0;
    `BI(pins, ALLOC_RENAME) = 0;
    `BI(pins, ALLOC_RESET_READ0) = 0;
    while (stale.size() != 0) begin
      `BI(pins, RECLAIM_VALID) = 1;
      `BI(pins, RECLAIM_BITS)  = stale.pop_front();
      tick();
    end
    `BI(pins, RECLAIM_VALID) = 0;
  endtask
  task body();
    restart();
    for (int n = 0; n < A - V; n++) allocate(n % V, 1, 0, 0, n % V, (n + 1) % V, (n + 2) % V);
    `BI(pins, ALLOC_RENAME) = 1;
    tick();
    evidence::check(!`BO(observed, READY), "BAT free list did not exhaust");
    reclaim_all();
    for (int n = 0; n < 4 * V; n++) begin
      allocate(n % V, 1, 0, 0, (n + 1) % V, (n + 2) % V, n % V);
      reclaim_all();
    end
    restart();
    for (int bank = 0; bank < V; bank++) allocate(bank, 1, 0, 0, bank, bank, bank);
    reclaim_all();
    for (int mask = 0; mask < 8; mask++)
      for (int read0 = 0; read0 < V; read0++)
        for (int read1 = 0; read1 < V; read1++)
          for (int read_wr = 0; read_wr < V; read_wr++)
            allocate(read0, 0, 0, 1, read0, read1, read_wr, mask);
    restart();
    allocate(0, 1, 0, 0, 0, 0, 0);
    allocate(0, 0, 1, 0, 1 % V, 0, 0);
    reclaim_all();
    restart();
    allocate(0, 1, 0, 0, 0, 0, 0);
    allocate(1 % V, 1, 0, 0, 1 % V, 0, 0);
    allocate(0, 0, 1, 0, 1 % V, 0, 0);
    reclaim_all();
    restart();
    allocate(0, 1, 0, 0, 0, 0, 0);
    allocate(0, 0, 1, 1, 0, 0, 0);
    reclaim_all();
    // A retired alias becomes available on the next cycle, alongside an independent allocation.
    restart();
    for (int n = 0; n < A - V - 1; n++) allocate(n % V, 1, 0, 0, n % V, n % V, n % V);
    while (stale.size() && stale[0] < V) begin
      `BI(pins, RECLAIM_VALID) = 1;
      `BI(pins, RECLAIM_BITS)  = stale.pop_front();
      tick();
    end
    `BI(pins, RECLAIM_VALID) = 1;
    `BI(pins, RECLAIM_BITS)  = stale.pop_front();
    allocate(0, 1, 0, 0, 0, 0, 0);
    `BI(pins, RECLAIM_VALID) = 0;
    reclaim_all();
    restart();
  endtask
endclass
class invalid_sequence extends scenario_sequence;
  `uvm_object_utils(invalid_sequence)
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task body();
    string scenario;
    if (!$value$plusargs("CASE=%s", scenario))
      `uvm_fatal("CASE", "BAT assertion scenario is required")
    restart();
    if (scenario == "full") begin
      for (int n = 0; n <= A - V; n++) allocate(0, 1, 0, 0, 0, 0, 0);
    end else if (scenario == "reclaim") begin
      `BI(pins, RECLAIM_VALID) = 1;
      `BI(pins, RECLAIM_BITS)  = A;
      tick();
    end else begin
      `BI(pins, ALLOC_VALID) = 1;
      case (scenario)
        "read0": begin
          `BI(pins, ALLOC_RAW_RD_BANK_0_VALID) = 1;
          `BI(pins, ALLOC_RAW_RD_BANK_0_ID) = V;
        end
        "read1": begin
          `BI(pins, ALLOC_RAW_RD_BANK_1_VALID) = 1;
          `BI(pins, ALLOC_RAW_RD_BANK_1_ID) = V;
        end
        "update": begin
          `BI(pins, ALLOC_RAW_RD_WR_VALID) = 1;
          `BI(pins, ALLOC_RAW_RD_WR_ID) = V;
        end
        "write": begin
          `BI(pins, ALLOC_RAW_WR_BANK_VALID) = 1;
          `BI(pins, ALLOC_RAW_WR_BANK_ID) = V;
        end
        default: `uvm_fatal("CASE", "Unknown BAT assertion scenario")
      endcase
      tick();
    end
    tick(3);
    `uvm_fatal("EXPECTED_ASSERTION", "Invalid BAT input did not trigger its assertion")
  endtask
endclass
