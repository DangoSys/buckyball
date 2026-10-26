class agent extends uvm_agent;
  `uvm_component_utils(agent)
  driver drv;
  monitor mon;
  uvm_sequencer #(cycle_item) seqr;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    drv  = driver::type_id::create("driver", this);
    mon  = monitor::type_id::create("monitor", this);
    seqr = new("seqr", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
  endfunction
endclass
