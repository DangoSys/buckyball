class mxint8mm_ball_test extends uvm_test;
  `uvm_component_utils(mxint8mm_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  mxint8mm_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
        mxint8mm_cycles::get_type());
    env = mxint8mm_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    for (int unsigned index = 0; index < mxint8mm_num_cases(); index++) run_case(index);
    phase.drop_objection(this);
  endtask

  // Steps of one case run back to back on the same reset.
  task run_case(int unsigned index);
    int unsigned steps;
    apply_reset();
    steps = mxint8mm_case_load(index, 1 << `BB_BANK_ADDR_W, 1 << `BB_ROB_ID_W);
    for (int unsigned step = 0; step < steps; step++) run_step(index, step);
  endtask

  task run_step(int unsigned index, int unsigned step);
    mxint8mm_step_seq seq;
    mxint8mm_cmd_dpi_t cmd;
    int cycles = 0;
    env.scb.reset_counters();
    mxint8mm_case_cmd(step, cmd);
    if (cmd.stall) begin
      @(negedge vif.clock);
      vif.cmd_resp_ready = 1'b0;
    end
    seq = mxint8mm_step_seq::type_id::create("seq");
    seq.step = step;
    seq.start(env.cmd_agent.seqr);
    if (cmd.stall) begin
      while (!vif.cmd_resp_valid) begin
        @(posedge vif.clock);
        cycles++;
        if (cycles > MXINT8MM_TIMEOUT_CYCLES)
          `uvm_fatal("TIMEOUT", $sformatf("case %0d step %0d: no response", index, step))
      end
      // The response stays valid with its ROB id while it is held back.
      repeat (MXINT8MM_STALL_CYCLES) begin
        @(posedge vif.clock);
        cycles++;
        if (vif.cmd_resp_valid !== 1'b1 || vif.cmd_resp_bits_rob_id !== cmd.rob_id[BB_BLINK_ROB_ID_W-1:0])
          `uvm_fatal("STALL", "held response changed")
        verification_pkg::evidence::checks++;
      end
      @(negedge vif.clock);
      vif.cmd_resp_ready = 1'b1;
    end
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > MXINT8MM_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("Mxint8mmBall case %0d step %0d timeout", index, step))
    end
    `uvm_info("MXINT8MM", $sformatf("case %0d step %0d M %0d N %0d K %0d passed in %0d cycles",
                                    index, step, cmd.rs2_lo[9:0], cmd.rs2_lo[19:10], cmd.iter,
                                    cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
