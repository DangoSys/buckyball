class transpose_ball_test extends uvm_test;
  `uvm_component_utils(transpose_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  transpose_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    end
    env = transpose_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    run_directed();
    run_random();
    phase.drop_objection(this);
  endtask

  task run_directed();
    run_case(0, "DIR_I8");
    run_case(1, "DIR_I32");
    run_case(2, "DIR_I8_2G");
  endtask

  task run_random();
    int i;
    for (i = 3; i <= 22; i++) begin
      run_case(i, $sformatf("RND_%0d", i));
    end
  endtask

  task run_case(int unsigned idx, string label);
    transpose_basic_seq seq;
    int cycles;

    apply_reset();
    env.scb.reset_counters();

    seq = transpose_basic_seq::type_id::create("seq");
    seq.case_index = idx;
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > TRANSPOSE_TIMEOUT_CYCLES) begin
        `uvm_fatal("TIMEOUT", $sformatf(
                                  "TransposeBall %s (idx=%0d) did not complete after %0d cycles",
                                  label, idx, TRANSPOSE_TIMEOUT_CYCLES))
      end
    end

    `uvm_info("TRANSPOSE", $sformatf("TransposeBall %s (idx=%0d) passed in %0d cycles", label, idx,
                                     cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
