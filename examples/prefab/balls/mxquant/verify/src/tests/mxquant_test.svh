class mxquant_ball_test extends uvm_test;
  `uvm_component_utils(mxquant_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  mxquant_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    env = mxquant_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < mxquant_num_cases(); index++) run_case(index);
    phase.drop_objection(this);
  endtask

  // Steps of one case run back to back on the same reset.
  task run_case(int unsigned index);
    int unsigned steps;
    apply_reset();
    steps = mxquant_case_load(index);
    for (int unsigned step = 0; step < steps; step++) run_step(index, step);
  endtask

  task run_step(int unsigned index, int unsigned step);
    mxquant_step_seq seq;
    int cycles = 0;
    env.scb.reset_counters();
    seq = mxquant_step_seq::type_id::create("seq");
    seq.step = step;
    seq.start(env.cmd_agent.seqr);
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > MXQUANT_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("MxquantBall case %0d step %0d timeout", index, step))
    end
    `uvm_info("MXQUANT", $sformatf("MxquantBall case %0d step %0d passed in %0d cycles", index,
                                   step, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
