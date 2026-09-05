class lut_ball_test extends uvm_test;
  `uvm_component_utils(lut_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t   vif;
  lut_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    env = lut_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned bid = lut_require_bid();
    phase.raise_objection(this);
    for (int unsigned index = 0; index < LUT_NUM_CASES; index++) run_case(index, bid);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned index, int unsigned bid);
    lut_basic_seq seq;
    int cycles = 0;
    apply_reset();
    env.scb.reset_counters();
    seq = lut_basic_seq::type_id::create("seq");
    seq.case_index = index;
    seq.bid = bid;
    seq.start(env.cmd_agent.seqr);
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > LUT_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("LutBall case %0d timeout", index))
    end
    `uvm_info("LUT", $sformatf("LutBall case %0d passed in %0d cycles", index, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
