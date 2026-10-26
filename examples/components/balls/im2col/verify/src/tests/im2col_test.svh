class im2col_ball_test extends uvm_test;
  `uvm_component_utils(im2col_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  im2col_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    end
    env = im2col_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < IM2COL_NUM_CASES; index++) run_case(index);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned idx);
    im2col_basic_seq seq;
    int cycles;

    apply_reset();
    env.scb.reset_counters();

    seq = im2col_basic_seq::type_id::create("seq");
    seq.case_index = idx;
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > IM2COL_TIMEOUT_CYCLES) begin
        `uvm_fatal("TIMEOUT",
                   $sformatf("Im2colBall case %0d timeout: cmds=%0d reads=%0d writes=%0d resp=%0d",
                             idx, env.scb.cmd_count, env.scb.read_count, env.scb.write_count,
                             env.scb.resp_count))
      end
    end

    `uvm_info("IM2COL", $sformatf("Im2colBall case %0d passed in %0d cycles", idx, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
