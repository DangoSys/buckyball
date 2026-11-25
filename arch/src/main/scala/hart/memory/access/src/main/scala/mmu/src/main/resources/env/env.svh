class env extends uvm_env;
  `uvm_component_utils(env)
  driver source;
  monitor mon;
  peer_driver peer;
  model reference;
  reset_controller reset;
  uvm_sequencer #(request_item) seqr;
  result_board #(result_item, 0) scoreboard;
  uvm_analysis_imp #(bit, env) reset_export;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_export = new("reset_export", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    mon = monitor::type_id::create("monitor", this);
    peer = peer_driver::type_id::create("peer", this);
    reference = model::type_id::create("model", this);
    reset = reset_controller::type_id::create("reset", this);
    seqr = new("seqr", this);
    scoreboard = result_board#(result_item, 0)::type_id::create("scoreboard", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    source.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(reference.analysis_export);
    peer.reference = reference;
    reference.expected_ap.connect(scoreboard.expected_export);
    reference.actual_ap.connect(scoreboard.actual_export);
    reference.reset_ap.connect(reset_export);
  endfunction
  function void write(bit value);
    scoreboard.cancel_pending();
  endfunction
endclass
