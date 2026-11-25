class agent extends uvm_agent;
  `uvm_component_utils(agent)
  uvm_sequencer #(request_item) seqr;
  driver drv;
  monitor mon;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    seqr = uvm_sequencer#(request_item)::type_id::create("seqr", this);
    drv  = driver::type_id::create("driver", this);
    mon  = monitor::type_id::create("monitor", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
  endfunction
endclass
