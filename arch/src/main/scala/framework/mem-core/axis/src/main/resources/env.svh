class env extends uvm_env;
  `uvm_component_utils(env)

  source_agent source;
  sink sink_driver;
  monitor sink_monitor;
  ref_model model;
  in_order_scoreboard #(item) scoreboard;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    virtual axis_if source_vif;
    virtual axis_if sink_vif;

    super.build_phase(phase);
    if (!uvm_config_db#(virtual axis_if)::get(this, "", "source_vif", source_vif)) begin
      `uvm_fatal("AXIS_VIF", "environment requires source_vif")
    end
    if (!uvm_config_db#(virtual axis_if)::get(this, "", "sink_vif", sink_vif)) begin
      `uvm_fatal("AXIS_VIF", "environment requires sink_vif")
    end
    uvm_config_db#(virtual axis_if)::set(this, "source.*", "vif", source_vif);
    uvm_config_db#(virtual axis_if)::set(this, "sink_driver", "vif", sink_vif);
    uvm_config_db#(virtual axis_if)::set(this, "sink_monitor", "vif", sink_vif);

    source       = source_agent::type_id::create("source", this);
    sink_driver  = sink::type_id::create("sink_driver", this);
    sink_monitor = monitor::type_id::create("sink_monitor", this);
    model        = ref_model::type_id::create("model", this);
    scoreboard   = in_order_scoreboard#(item)::type_id::create("scoreboard", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    source.mon.ap.connect(model.analysis_export);
    model.expected_ap.connect(scoreboard.expected_export);
    sink_monitor.ap.connect(scoreboard.actual_export);
  endfunction
endclass
