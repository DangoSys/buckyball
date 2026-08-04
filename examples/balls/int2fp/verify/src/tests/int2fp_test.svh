class int2fp_ball_test extends uvm_test;
  `uvm_component_utils(int2fp_ball_test)

  typedef virtual bb_blink_if #(1, 1) vif_t;
  vif_t vif;
  int2fp_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    end
    env = int2fp_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned bid;
    bid = int2fp_require_bid();
    phase.raise_objection(this);
    run_directed(bid);
    run_random(bid);
    phase.drop_objection(this);
  endtask

  task run_directed(int unsigned bid);
    run_case(0, "DIR_FP32", bid);
    run_case(1, "DIR_I8", bid);
  endtask

  task run_random(int unsigned bid);
    int i;
    for (i = 2; i <= 21; i++) begin
      run_case(i, $sformatf("RND_%0d", i), bid);
    end
  endtask

  task run_case(int unsigned idx, string label, int unsigned bid);
    int2fp_basic_seq seq;
    int cycles;

    apply_reset();
    env.scb.reset_counters();

    seq = int2fp_basic_seq::type_id::create("seq");
    seq.case_index = idx;
    seq.bid = bid;
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > INT2FP_TIMEOUT_CYCLES) begin
        `uvm_fatal(
            "TIMEOUT",
            $sformatf(
                "Int2FpBall %s (idx=%0d) timeout: cmds=%0d reads=%0d/%0d writes=%0d/%0d resp=%0d",
                label, idx, env.scb.cmd_count, env.scb.read_count, env.scb.expected_reads,
                env.scb.write_count, env.scb.expected_writes, env.scb.resp_count))
      end
    end

    `uvm_info("INT2FP", $sformatf("Int2FpBall %s (idx=%0d) passed in %0d cycles", label, idx,
                                  cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
