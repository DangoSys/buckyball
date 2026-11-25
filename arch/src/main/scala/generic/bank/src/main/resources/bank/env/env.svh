`uvm_analysis_imp_decl(_reset)

class env extends environment #(request_item, response_item, ref_model);
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
    bus = agent::type_id::create("agent", this);
    cov = coverage::type_id::create("coverage", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    bus.mon.request_ap.connect(input_export);
    bus.mon.request_ap.connect(cov.analysis_export);
    bus.mon.response_ap.connect(output_export);
    bus.mon.reset_ap.connect(reset_export);
  endfunction

  function void write_reset(bit asserted);
    // Reset cancels responses; accepted writes remain in the reference memory.
    scoreboard.cancel_pending();
  endfunction
endclass
