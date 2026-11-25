class mxint8mm_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(mxint8mm_env)

  mxint8mm_scoreboard scb;
  mxint8mm_cov icov;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    scb  = mxint8mm_scoreboard::type_id::create("scb", this);
    icov = mxint8mm_cov::type_id::create("icov", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    scb.mem_model = mem_model;
    cmd_agent.stim_ap.connect(scb.stim_imp);
    cmd_agent.stim_ap.connect(icov.stim_imp);
    cmd_agent.cmd_ap.connect(scb.cmd_imp);
    read_mon.read_ap.connect(scb.read_imp);
    write_mon.write_ap.connect(scb.write_imp);
    resp_mon.resp_ap.connect(scb.resp_imp);
  endfunction
endclass
