class env extends uvm_env;
  `uvm_component_utils(env)
  uvm_analysis_imp #(uvm_sequence_item, env) reset_events;
  driver source;
  monitor mon;
  peer backend;
  model reference;
  uvm_sequencer #(cycle_item) seqr;
  result_board #(result_item, 0) scoreboard;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_events = new("reset_events", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    mon = monitor::type_id::create("monitor", this);
    backend = peer::type_id::create("backend", this);
    reference = model::type_id::create("model", this);
    seqr = new("seqr", this);
    scoreboard = result_board#(result_item, 0)::type_id::create("scoreboard", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    fixture image = fixture::type_id::create("image");
    reference.image = image;
    backend.image   = image;
    source.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(reference.analysis_export);
    reference.expected.connect(scoreboard.expected_export);
    reference.actual.connect(scoreboard.actual_export);
    reference.cancelled.connect(reset_events);
  endfunction
  function void write(uvm_sequence_item t);
    scoreboard.cancel_pending();
  endfunction
endclass
