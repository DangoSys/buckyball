class env extends uvm_env;
  `uvm_component_utils(env)
  driver drv;
  monitor mon;
  reference_model reference;
  uvm_sequencer #(cycle_item) seqr;
  result_board #(result_item, 0) scoreboard;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    drv = driver::type_id::create("driver", this);
    mon = monitor::type_id::create("monitor", this);
    reference = reference_model::type_id::create("model", this);
    seqr = new("seqr", this);
    scoreboard = result_board#(result_item, 0)::type_id::create("scoreboard", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(reference.analysis_export);
    reference.expected_ap.connect(scoreboard.expected_export);
    reference.actual_ap.connect(scoreboard.actual_export);
  endfunction
endclass
