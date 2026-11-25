class env extends environment #(operation, item, ref_model);
  `uvm_component_utils(env)
  command_agent commands;
  axis_source_agent #(32) source;
  axis_sink #(32) sink;
  axis_monitor #(32) output_monitor;
  reset_controller reset;
  coverage cov;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    commands = command_agent::type_id::create("commands", this);
    source = axis_source_agent#(32)::type_id::create("source", this);
    sink = axis_sink#(32)::type_id::create("sink", this);
    output_monitor = axis_monitor#(32)::type_id::create("output_monitor", this);
    reset = reset_controller::type_id::create("reset", this);
    cov = coverage::type_id::create("coverage", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    commands.monitor.ap.connect(input_export);
    commands.monitor.commands.connect(cov.analysis_export);
    output_monitor.ap.connect(output_export);
  endfunction
endclass
