class mxint8quant_ball_test extends uvm_test;
  `uvm_component_utils(mxint8quant_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  mxint8quant_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
        mxint8quant_cycles::get_type());
    env = mxint8quant_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < mxint8quant_num_cases(); index++) run_case(index);
    phase.drop_objection(this);
  endtask

  // Steps of one case run back to back on the same reset.
  task run_case(int unsigned index);
    int unsigned steps;
    apply_reset();
    steps = mxint8quant_case_load(index);
    for (int unsigned step = 0; step < steps; step++) run_step(index, step);
  endtask

  task run_step(int unsigned index, int unsigned step);
    mxint8quant_step_seq seq;
    int unsigned hold;
    int cycles = 0;
    env.scb.reset_counters();
    hold = mxint8quant_case_hold(step);
    if (hold != 0) begin
      @(negedge vif.clock);
      vif.cmd_resp_ready = 1'b0;
    end
    seq = mxint8quant_step_seq::type_id::create("seq");
    seq.step = step;
    seq.start(env.cmd_agent.seqr);
    if (hold != 0) begin
      while (!vif.cmd_resp_valid) tick(index, step, cycles);
      // The response stays valid and unchanged until it is taken.
      repeat (hold) begin
        tick(index, step, cycles);
        if (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id !== env.scb.stim.rob_id ||
            env.scb.resp_count != 0)
          `uvm_fatal("HOLD", "held response did not stay valid and stable")
      end
      @(negedge vif.clock);
      vif.cmd_resp_ready = 1'b1;
    end
    while (!env.scb.done()) tick(index, step, cycles);
    `uvm_info("MXINT8QUANT",
              $sformatf("case %0d step %0d rows %0d K %0d columns %0d passed in %0d cycles", index,
                        step, env.scb.stim.iter, env.scb.stim.rs2[15:0], env.scb.stim.rs2[44],
                        cycles), UVM_LOW)
  endtask

  task tick(int unsigned index, int unsigned step, inout int cycles);
    @(posedge vif.clock);
    cycles++;
    if (cycles > MXINT8QUANT_TIMEOUT_CYCLES)
      `uvm_fatal("TIMEOUT", $sformatf("Mxint8quantBall case %0d step %0d timeout", index, step))
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
