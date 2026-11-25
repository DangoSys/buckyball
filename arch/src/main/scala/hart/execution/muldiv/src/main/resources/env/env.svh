`uvm_analysis_imp_decl(_reset)
class env extends environment #(request_item, response_item, model);
  `uvm_component_utils(env)
  agent bus;
  coverage cov;
  uvm_analysis_imp_reset #(bit, env) reset_export;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_export = new("reset_export", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    cov = coverage::type_id::create("coverage", this);
    bus = agent::type_id::create("agent", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    bus.mon.requests.connect(input_export);
    bus.mon.requests.connect(cov.analysis_export);
    bus.mon.reset_ap.connect(reset_export);
    bus.mon.responses.connect(output_export);
  endfunction
  function void write_reset(bit asserted);
    scoreboard.cancel_pending();
  endfunction
endclass
