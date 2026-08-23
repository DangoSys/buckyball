class smatmul_ball_test extends uvm_test;
  `uvm_component_utils(smatmul_ball_test)

  typedef virtual bb_blink_if #(2, 4) vif_t;
  vif_t vif;
  smatmul_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    end
    env = smatmul_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned bid;
    bid = matrix_require_bid();
    phase.raise_objection(this);
    run_directed(bid);
    run_random(bid);
    phase.drop_objection(this);
  endtask

  task run_directed(int unsigned bid);
    run_case(0, "DIR_OS_4x4", bid);
    run_case(1, "DIR_OS_5x7x3", bid);
    run_case(2, "DIR_OS_16x16", bid);
    run_case(3, "DIR_WS_32x16", bid);
  endtask

  task run_random(int unsigned bid);
    int i;
    for (i = 4; i <= 23; i++) begin
      run_case(i, $sformatf("RND_%0d", i), bid);
    end
  endtask

  task run_case(int unsigned idx, string label, int unsigned bid);
    matrix_basic_seq seq;
    int cycles;
    apply_reset();
    env.scb.reset_counters();

    seq = matrix_basic_seq::type_id::create("seq");
    seq.case_index = idx;
    seq.bid = bid;
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > MATRIX_TIMEOUT_CYCLES) begin
        `uvm_fatal(
            "TIMEOUT",
            $sformatf(
                "SMatMulBall %s (idx=%0d) timeout: cmds=%0d r0=%0d r1=%0d wr=%0d/%0d resp=%0d",
                label, idx, env.scb.cmd_count, env.scb.read_count[0], env.scb.read_count[1],
                env.scb.write_count, env.scb.expected_writes, env.scb.resp_count))
      end
    end

    `uvm_info("MATRIX", $sformatf("SMatMulBall %s (idx=%0d) passed in %0d cycles", label, idx,
                                  cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    @(posedge vif.clock);
    repeat (4) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
