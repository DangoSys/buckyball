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
    fp2int_basic_seq seq;
    int cycles;

    phase.raise_objection(this);

    apply_reset();

    seq = fp2int_basic_seq::type_id::create("seq");
    seq.start(env.cmd_agent.seqr);

    cycles = 0;
    while (!env.scb.done()) begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > FP2INT_TIMEOUT_CYCLES) begin
        `uvm_fatal("TIMEOUT", "Fp2IntBall test did not complete")
      end
    end

    `uvm_info("FP2INT", "Fp2IntBall INT32 case passed", UVM_LOW)
    phase.drop_objection(this);
  endtask

  task apply_reset();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
