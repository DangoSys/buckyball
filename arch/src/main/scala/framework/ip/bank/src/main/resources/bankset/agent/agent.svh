class command_agent extends uvm_agent;
  `uvm_component_utils(command_agent)
  uvm_sequencer #(command) seqr;
  command_driver driver;
  command_monitor monitor;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    seqr = new("seqr", this);
    driver = command_driver::type_id::create("driver", this);
    monitor = command_monitor::type_id::create("monitor", this);
  endfunction
  function void connect_phase(uvm_phase phase);
    driver.seq_item_port.connect(seqr.seq_item_export);
  endfunction
endclass
