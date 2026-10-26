class env extends environment #(item, item, ref_model);
  `uvm_component_utils(env)

  coverage coverage_monitor;
  source_agent source;
  sink sink_driver;
  monitor sink_monitor;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    virtual axis_if #(`AXIS_DATA_BITS) source_vif;
    virtual axis_if #(`AXIS_DATA_BITS) sink_vif;

    super.build_phase(phase);
    if (!uvm_config_db#(virtual axis_if #(`AXIS_DATA_BITS))::get(
            this, "", "source_vif", source_vif
        )) begin
      `uvm_fatal("AXIS_VIF", "environment requires source_vif")
    end
    if (!uvm_config_db#(virtual axis_if #(`AXIS_DATA_BITS))::get(
            this, "", "sink_vif", sink_vif
        )) begin
      `uvm_fatal("AXIS_VIF", "environment requires sink_vif")
    end
    uvm_config_db#(virtual axis_if #(`AXIS_DATA_BITS))::set(this, "source.*", "vif", source_vif);
    uvm_config_db#(virtual axis_if #(`AXIS_DATA_BITS))::set(this, "sink_driver", "vif", sink_vif);
    uvm_config_db#(virtual axis_if #(`AXIS_DATA_BITS))::set(this, "sink_monitor", "vif", sink_vif);

    coverage_monitor = coverage::type_id::create("coverage", this);
    source           = source_agent::type_id::create("source", this);
    sink_driver      = sink::type_id::create("sink_driver", this);
    sink_monitor     = monitor::type_id::create("sink_monitor", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    source.mon.ap.connect(input_export);
    sink_monitor.ap.connect(output_export);
    sink_monitor.ap.connect(coverage_monitor.analysis_export);
  endfunction
endclass
