class env extends uvm_env;
  `uvm_component_utils(env)
  agent bus;
  protocol_model state;
  reset_controller reset;
  result_board #(result_item, 0) scoreboard;
  uvm_analysis_imp #(bit, env) reset_export;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_export = new("reset_export", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    state = protocol_model::type_id::create("model", this);
    bus   = agent::type_id::create("bus", this);
    reset = reset_controller::type_id::create("reset", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    bus.mon.ap.connect(state.analysis_export);
    bus.endpoint.state = state;
    bus.source0.state  = state;
    bus.source1.state  = state;
    bus.source2.state  = state;
  endfunction
  function void write(bit value);
    scoreboard.cancel_pending();
  endfunction
endclass
