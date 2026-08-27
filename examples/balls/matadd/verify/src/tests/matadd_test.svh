class matadd_ball_test extends uvm_test;
  `uvm_component_utils(matadd_ball_test)

  typedef virtual bb_blink_if #(2, 1) vif_t;
  vif_t vif;
  matadd_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    env = matadd_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned bid;
    bid = matadd_require_bid();
    phase.raise_objection(this);
    run_case(0, "ONE_GROUP_16_LINES", bid);
    run_case(1, "TWO_GROUPS_16_LINES", bid);
    run_case(2, "TWO_GROUPS_1024_LINES", bid);
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned index, string name, int unsigned bid);
    matadd_seq seq;
    int cycles;
    apply_reset();
    env.scb.reset_counters();
    seq = matadd_seq::type_id::create("seq");
    seq.case_index = index;
    seq.bid = bid;
    seq.start(env.cmd_agent.seqr);
    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > MATADD_TIMEOUT_CYCLES)
        `uvm_fatal("TIMEOUT", $sformatf(
                   "%s timed out reads=%0d/%0d writes=%0d resp=%0d",
                   name,
                   env.scb.read_count[0],
                   env.scb.read_count[1],
                   env.scb.write_count,
                   env.scb.resp_count
                   ))
    end
    `uvm_info("MATADD", $sformatf("%s passed in %0d cycles", name, cycles), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    @(posedge vif.clock);
    repeat (4) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
