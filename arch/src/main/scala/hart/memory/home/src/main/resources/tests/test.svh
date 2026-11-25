class home_test extends protocol_test;
  `uvm_component_utils(home_test)
  transport_monitor transport;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    transport = transport_monitor::type_id::create("transport", this);
  endfunction
  task execute();
    home_sequence seq = home_sequence::type_id::create("scenario");
    seq.test_env = test_env;
    seq.control  = control;
    seq.start(test_env.bus.seqr);
  endtask
endclass
