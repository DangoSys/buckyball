class trace_ball_test extends uvm_test;
  `uvm_component_utils(trace_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  trace_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cov#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(trace_blink_cov::get_type());
    env = trace_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < TRACE_NUM_CASES; index++) run_case(index);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned index);
    trace_basic_seq seq;
    int cycles = 0;
    apply_reset();
    seq = trace_basic_seq::type_id::create("seq");
    seq.num_cmds = trace_case_load(index);
    env.scb.reset_counters(seq.num_cmds);
    seq.start(env.cmd_agent.seqr);
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > TRACE_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("TraceBall case %0d timeout", index))
    end
    // The trace hook runs one cycle after the command; let it drain.
    repeat (2) @(posedge vif.clock);
    env.scb.check_records();
    `uvm_info("TRACE", $sformatf("TraceBall case %0d passed: %0d commands in %0d cycles", index,
                                 seq.num_cmds, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
