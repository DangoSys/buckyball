// Illegal input: SMATMUL_OS with M = 24, not one and not a multiple of
// tileRows. The command assert "SMATMUL_OS M must be one or a positive
// multiple of tileRows" must reject it on the command handshake; BEMU panics
// on the same input. The test passes only when that RTL assertion fires: the
// run expects exactly one assertion failure with that message and no response.
class smatmul_m24_test extends smatmul_latency_test;
  `uvm_component_utils(smatmul_m24_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    smatmul_latency_seq seq;
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 4; row++) env.mem_model.load_word_g(4, 0, row, '0);
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
    end
    env.mem_model.arm();
    measure("SMATMUL_BIAS", 1, 0, 0, 0, 0, 0, 0, 0);
    seq = smatmul_latency_seq::type_id::create("seq");
    seq.m = 24;
    seq.k = 16;
    seq.first = 1;
    seq.last = 1;
    seq.k_groups = 1;
    seq.wr_col = 1;
    seq.c_base = 0;
    seq.start(env.cmd_agent.seqr);
    repeat (2) @(posedge vif.clock);
    phase.drop_objection(this);
  endtask
endclass
