class env extends environment #(cycle_item, result_item, ref_model);
  `uvm_component_utils(env)
  agent bus;
  coverage cov;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    bus = agent::type_id::create("agent", this);
    cov = coverage::type_id::create("coverage", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    bus.mon.input_ap.connect(input_export);
    bus.mon.input_ap.connect(cov.analysis_export);
    bus.mon.output_ap.connect(output_export);
  endfunction
endclass
