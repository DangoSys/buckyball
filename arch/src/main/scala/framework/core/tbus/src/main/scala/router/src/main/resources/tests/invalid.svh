class invalid_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(invalid_sequence)
  logic [`TR_INPUT_BITS-1:0] pins;
  function new(string name = "invalid");
    super.new(name);
  endfunction
  task tick(bit reset = 0);
    cycle_item t = cycle_item::type_id::create("cycle");
    start_item(t);
    t.inputs = pins;
    t.reset  = reset;
    finish_item(t);
  endtask
  task body();
    string scenario;
    if (!$value$plusargs("CASE=%s", scenario)) `uvm_fatal("CASE", "Invalid test requires a case")
    pins = '0;
    repeat (3) tick(1);
    tick();
    if (scenario == "physical") begin
      pins[`TR_IN_PHYSICAL_0_REQUEST_VALID_OFFSET] = 1;
      pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_ADDRESS_OFFSET+:64] = 1;
    end
    if (scenario == "lease") begin
      pins[`TR_IN_LEASE_REQUEST_VALID_OFFSET] = 1;
      pins[`TR_IN_LEASEOWNERTHREADID_OFFSET+:64] = 10;
    end
    if (scenario == "transfer") begin
      pins[`TR_IN_CONFIG_VALID_OFFSET] = 1;
      pins[`TR_IN_CONFIG_BITS_TRANSFER_OFFSET] = 1;
      pins[`TR_IN_CONFIG_BITS_THREAD_ID_OFFSET+:64] = 10;
      pins[`TR_IN_CONFIG_BITS_SOURCE_BANK_ID_OFFSET+:`TR_IN_CONFIG_BITS_SOURCE_BANK_ID_WIDTH] = 1;
    end
    if (scenario == "alias") begin
      pins[`TR_IN_MAPPING_0_VALID_OFFSET] = 1;
      pins[`TR_IN_MAPPING_0_BITS_ALLOCATE_OFFSET] = 1;
      pins[`TR_IN_MAPPING_0_BITS_THREADID_OFFSET+:64] = 10;
      pins[`TR_IN_MAPPING_0_BITS_VBANK_OFFSET+:`TR_IN_MAPPING_0_BITS_VBANK_WIDTH] = 6;
    end
    repeat (10) tick();
    `uvm_fatal("MISSING_ASSERTION", "Expected Router assertion did not fire")
  endtask
endclass
class invalid_test extends base_test;
  `uvm_component_utils(invalid_test)
  driver source;
  uvm_sequencer #(cycle_item) seqr;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    seqr   = new("seqr", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    source.seq_item_port.connect(seqr.seq_item_export);
  endfunction
  task execute();
    invalid_sequence seq = invalid_sequence::type_id::create("invalid");
    seq.start(seqr);
  endtask
endclass
