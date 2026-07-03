class fp2int_cmd_agent extends uvm_agent;
  `uvm_component_utils(fp2int_cmd_agent)

  uvm_sequencer #(fp2int_cmd_item) seqr;
  fp2int_cmd_driver drv;
  fp2int_cmd_monitor mon;
  uvm_analysis_port #(fp2int_cmd_item) stim_ap;
  uvm_analysis_port #(fp2int_cmd_item) cmd_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_ap = new("stim_ap", this);
    cmd_ap  = new("cmd_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    seqr = uvm_sequencer#(fp2int_cmd_item)::type_id::create("seqr", this);
    drv  = fp2int_cmd_driver::type_id::create("drv", this);
    mon  = fp2int_cmd_monitor::type_id::create("mon", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
    drv.stim_ap.connect(stim_ap);
    mon.cmd_ap.connect(cmd_ap);
  endfunction
endclass
