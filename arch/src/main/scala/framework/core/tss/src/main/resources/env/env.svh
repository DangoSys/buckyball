class env extends uvm_env;
  `uvm_component_utils(env)
  driver source;
  monitor mon;
  model reference;
  uvm_sequencer #(cycle_item) seqr;
  result_board #(result_item, 0) boards[`TSS_PORTS];
  uvm_analysis_imp #(result_item, env) results;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    results = new("results", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    source = driver::type_id::create("source", this);
    mon = monitor::type_id::create("monitor", this);
    reference = model::type_id::create("model", this);
    seqr = new("seqr", this);
    foreach (boards[p])
    boards[p] = result_board#(result_item, 0)::type_id::create($sformatf("port%0d", p), this);
  endfunction
  function void connect_phase(uvm_phase phase);
    source.seq_item_port.connect(seqr.seq_item_export);
    mon.ap.connect(reference.analysis_export);
    reference.results.connect(results);
  endfunction
  function void write(result_item t);
    case (t.message_kind)
      0: boards[t.port].write_expected(t);
      1: boards[t.port].write_actual(t);
      2: boards[t.port].cancel_pending();
    endcase
  endfunction
endclass
