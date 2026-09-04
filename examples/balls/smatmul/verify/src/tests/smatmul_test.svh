class smatmul_ball_test extends uvm_test;
  `uvm_component_utils(smatmul_ball_test)

  typedef virtual bb_blink_if #(2, 1) vif_t;
  vif_t vif;
  smatmul_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    if (SMATMUL_OUT_BW != 1) `uvm_fatal("CONFIG", "current SMatMul contract requires outBW=1")
    env = smatmul_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned bid = matrix_require_bid();
    phase.raise_objection(this);
    for (int unsigned index = 0; index < 11; index++) run_case(index, bid);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned index, int unsigned bid);
    smatmul_chain_seq seq;
    int cycles = 0;
    apply_reset();
    env.scb.reset_counters();
    seq = smatmul_chain_seq::type_id::create("seq");
    seq.case_index = index;
    seq.bid = bid;
    seq.start(env.cmd_agent.seqr);
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > MATRIX_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf("case %0d timed out", index))
    end
    `uvm_info("SMATMUL", $sformatf("case %0d passed in %0d cycles", index, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
