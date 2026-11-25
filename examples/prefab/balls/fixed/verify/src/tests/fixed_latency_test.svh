// FIXED latency: one instruction per rs2 mode (and per streaming variant of
// the coefficient modes) with the fewest rows it takes, 4 (iter a positive
// multiple of four): their fixed cycles. Same data and bank model as
// fixed_throughput_test.
class fixed_latency_test extends fixed_perf_test;
  `uvm_component_utils(fixed_latency_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    run_one("requant", 0, 0, 4);
    run_one("hswish", 1, 0, 4);
    run_one("hsigmoid", 2, 0, 4);
    run_one("mul", 3, 0, 4);
    run_one("relu", 4, 0, 4);
    run_one("add", 5, 0, 4);
    run_one("sum", 6, 0, 4);
    run_one("mean", 7, 0, 4);
    run_one("requant_stream", 0, 1, 4);
    run_one("hswish_stream", 1, 1, 4);
    run_one("hsigmoid_stream", 2, 1, 4);
    phase.drop_objection(this);
  endtask
endclass
