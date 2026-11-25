class env extends uvm_env;
  `uvm_component_utils(env)
  driver drv;
  monitor mon;
  uvm_sequencer #(batch) seqr;
  routing_model model;
  result_board #(result_item, 0) scoreboard;
  reset_controller reset;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    drv = driver::type_id::create("driver", this);
    mon = monitor::type_id::create("monitor", this);
    seqr = new("seqr", this);
    model = routing_model::type_id::create("model", this);
    scoreboard = result_board#(result_item, 0)::type_id::create("scoreboard", this);
    reset = reset_controller::type_id::create("reset", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(model.analysis_export);
    model.expected_ap.connect(scoreboard.expected_export);
    model.actual_ap.connect(scoreboard.actual_export);
  endfunction
endclass
