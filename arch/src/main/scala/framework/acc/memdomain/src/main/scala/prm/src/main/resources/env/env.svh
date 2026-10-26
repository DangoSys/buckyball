class env extends uvm_env;
  `uvm_component_utils(env)
  driver drv;
  monitor mon;
  memory_model model;
  uvm_sequencer #(cycle_item) seqr;
  result_board #(result_item, 1) scoreboard;
  uvm_analysis_imp #(bit, env) reset_export;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    reset_export = new("reset_export", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    drv = driver::type_id::create("driver", this);
    mon = monitor::type_id::create("monitor", this);
    model = memory_model::type_id::create("model", this);
    seqr = new("seqr", this);
    scoreboard = result_board#(result_item, 1)::type_id::create("scoreboard", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(model.analysis_export);
    model.expected_ap.connect(scoreboard.expected_export);
    model.actual_ap.connect(scoreboard.actual_export);
    model.reset_ap.connect(reset_export);
  endfunction
  function void write(bit value);
    scoreboard.cancel_pending();
  endfunction
endclass
