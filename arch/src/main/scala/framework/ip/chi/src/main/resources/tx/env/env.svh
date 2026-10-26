class env extends uvm_env;
  `uvm_component_utils(env)
  agent bus;
  model reference;
  result_board #(flit_item, 0) scoreboard;
  uvm_analysis_imp #(bit, env) reset_export;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_export = new("reset_export", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    bus = agent::type_id::create("bus", this);
    reference = model::type_id::create("model", this);
    scoreboard = result_board#(flit_item, 0)::type_id::create("scoreboard", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    bus.mon.ap.connect(reference.analysis_export);
    reference.expected_ap.connect(scoreboard.expected_export);
    reference.actual_ap.connect(scoreboard.actual_export);
    reference.reset_ap.connect(reset_export);
  endfunction
  function void write(bit value);
    scoreboard.cancel_pending();
  endfunction
endclass
