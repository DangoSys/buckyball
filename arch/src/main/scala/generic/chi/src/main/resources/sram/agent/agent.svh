class agent extends uvm_agent;
  `uvm_component_utils(agent)
  monitor mon;
  endpoint_driver endpoint;
  uvm_sequencer #(packet_item) seqr[2];
  source_driver #(0) source0;
  source_driver #(1) source1;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    mon = monitor::type_id::create("monitor", this);
    endpoint = endpoint_driver::type_id::create("endpoint", this);
    seqr[0] = new("seqr0", this);
    source0 = source_driver#(0)::type_id::create("source0", this);
    seqr[1] = new("seqr1", this);
    source1 = source_driver#(1)::type_id::create("source1", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    source0.seq_item_port.connect(seqr[0].seq_item_export);
    source1.seq_item_port.connect(seqr[1].seq_item_export);
  endfunction
endclass
