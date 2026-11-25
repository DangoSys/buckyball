class cache_test extends protocol_test;
  `uvm_component_utils(cache_test)
  virtual cmo_if cmo;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    protocol_model::type_id::set_type_override(cache_model::get_type());
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cmo_if)::get(this, "", "cmo", cmo))
      `uvm_fatal("VIF", "Cache maintenance interface is required")
  endfunction
  task execute();
    cache_sequence seq = cache_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.state = test_env.state;
    seq.vif = vif;
    seq.scenario = test_env.state.scenario;
    seq.cmo = cmo;
    if (!$cast(seq.checked_state, test_env.state))
      `uvm_fatal("TYPE", "Maintenance model is required")
    if (seq.scenario == "normal") seq.start(null);
    else begin
      repeat (3) @(negedge vif.clock);
      test_env.reset.pulse(4);
      cmo.first_line = 'h40000;
      cmo.last_line = 'h40000;
      cmo.operation = 0;
      cmo.tag = 1;
      if (seq.scenario == "bad_maintenance_alignment") cmo.first_line = 'h40001;
      if (seq.scenario == "bad_maintenance_range") cmo.last_line = 'h3ffc0;
      if (seq.scenario == "bad_maintenance_operation") cmo.operation = 3;
      cmo.valid = 1;
      do @(posedge vif.clock); while (!cmo.ready);
      @(negedge vif.clock);
      cmo.valid = 0;
      repeat (40) @(negedge vif.clock);
      `uvm_fatal("MISSING_ASSERTION", "Expected maintenance assertion did not fire")
    end
  endtask
endclass
