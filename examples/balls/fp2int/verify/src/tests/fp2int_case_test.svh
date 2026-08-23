virtual class fp2int_case_test extends uvm_test;
  typedef virtual bb_blink_if #(1, 1) vif_t;
  vif_t vif;
  fp2int_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    env = fp2int_env::type_id::create("env", this);
  endfunction

  pure virtual function int unsigned case_index();
  pure virtual function string case_label();

  task run_phase(uvm_phase phase);
    fp2int_basic_seq seq;
    int cycles;

    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.scb.reset_counters();
    seq = fp2int_basic_seq::type_id::create("seq");
    seq.case_index = case_index();
    seq.bid = fp2int_require_bid();
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > FP2INT_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("Fp2IntBall %s timeout", case_label()))
    end
    `uvm_info("FP2INT", $sformatf("Fp2IntBall %s passed", case_label()), UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass
