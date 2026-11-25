class fixed_ball_test extends uvm_test;
  `uvm_component_utils(fixed_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  fixed_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(fixed_cycles::get_type());
    env = fixed_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < FIXED_NUM_CASES; index++) run_case(index);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned index);
    fixed_basic_seq seq;
    int cycles = 0;
    apply_reset();
    env.scb.reset_counters();
    seq = fixed_basic_seq::type_id::create("seq");
    seq.case_index = index;
    seq.start(env.cmd_agent.seqr);
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > FIXED_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("FixedBall case %0d timeout", index))
    end
    `uvm_info("FIXED", $sformatf("FixedBall case %0d passed in %0d cycles", index, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
