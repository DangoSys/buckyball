class fp2int_env extends uvm_env;
  `uvm_component_utils(fp2int_env)

  fp2int_cmd_agent cmd_agent;
  fp2int_mem_model mem_model;
  fp2int_read_monitor read_mon;
  fp2int_write_monitor write_mon;
  fp2int_resp_monitor resp_mon;
  fp2int_scoreboard scb;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    cmd_agent = fp2int_cmd_agent::type_id::create("cmd_agent", this);
    mem_model = fp2int_mem_model::type_id::create("mem_model", this);
    read_mon = fp2int_read_monitor::type_id::create("read_mon", this);
    write_mon = fp2int_write_monitor::type_id::create("write_mon", this);
    resp_mon = fp2int_resp_monitor::type_id::create("resp_mon", this);
    scb = fp2int_scoreboard::type_id::create("scb", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    cmd_agent.stim_ap.connect(mem_model.stim_imp);
    cmd_agent.stim_ap.connect(scb.stim_imp);
    cmd_agent.cmd_ap.connect(scb.cmd_imp);
    read_mon.read_ap.connect(scb.read_imp);
    write_mon.write_ap.connect(scb.write_imp);
    resp_mon.resp_ap.connect(scb.resp_imp);
  endfunction
endclass
