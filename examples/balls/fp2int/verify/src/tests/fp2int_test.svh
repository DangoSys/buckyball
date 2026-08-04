class fp2int_ball_test extends uvm_test;
  `uvm_component_utils(fp2int_ball_test)

  virtual fp2int_if vif;
  fp2int_env env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
    env = fp2int_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    run_case(0, "FP32_TO_INT8");
    phase.drop_objection(this);
  endtask

  task run_case(int unsigned case_index, string label);
    fp2int_basic_seq seq;
    int cycles;

    apply_reset();
    env.scb.stim_q.delete();
    env.scb.cmd_count = 0;
    env.scb.read_count = 0;
    env.scb.write_count = 0;
    env.scb.resp_count = 0;
    env.scb.expect_group = 0;

    seq = fp2int_basic_seq::type_id::create("seq");
    seq.case_index = case_index;
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > FP2INT_TIMEOUT_CYCLES) begin
        `uvm_fatal("TIMEOUT", $sformatf("Fp2IntBall %s case did not complete", label))
      end
    end

    `uvm_info("FP2INT", $sformatf("Fp2IntBall %s case passed", label), UVM_LOW)
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
