// FIXED throughput: one instruction per rs2 mode (and per streaming variant of
// the coefficient modes) with the most rows it takes: 1024 rows, the whole
// bank; streaming 512 (two coefficient rows per row); MEAN only takes 4.
// Every mode takes its own path through the FSM, so each is its own entry.
// The data drives the slowest per-lane path: A lanes are 2, every B lane is 1
// (multipliers 1 and shifts 1, so no multiply is skipped, q = 1 with
// fraction 1 lies strictly inside (-3Q, 3Q) and is nonzero, so HSWISH and
// HSIGMOID take the multiply and divide paths; MEAN divisors 1).
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// The shared cycle checker holds it, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); the BBPERF line reports its cycles.
class fixed_perf_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(fixed_perf_seq)

  bit [63:0] rs2;
  int unsigned iter;

  function new(string name = "fixed_perf_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = FIXED_THREAD_FUNCT7[6:0];
    item.rs2          = rs2;
    item.iter         = iter;
    item.op1_bank     = 1;
    item.op2_bank     = 2;
    item.wr_bank      = 3;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = 1;
    item.op2_col      = 1;
    item.wr_col       = 1;
    item.wr_read_bank = 0;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

// rs2: mode in [2:0], fraction in [8:4] (always 1), streaming in [9].
class fixed_perf_test extends uvm_test;
  `uvm_component_utils(fixed_perf_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(fixed_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  // One instruction on a fresh reset, then its cycles.
  task run_one(string label, int unsigned mode, bit stream, int unsigned iter);
    fixed_perf_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, {4{32'd2}});
      env.mem_model.load_word_g(2, 0, row, {4{32'd1}});
    end
    env.mem_model.arm();
    seq = fixed_perf_seq::type_id::create("seq");
    seq.rs2 = mode | (1 << 4) | (stream << 9);
    seq.iter = iter;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF FIXED.%s %0d %0d", label, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass

class fixed_throughput_test extends fixed_perf_test;
  `uvm_component_utils(fixed_throughput_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    run_one("requant", 0, 0, 1024);
    run_one("hswish", 1, 0, 1024);
    run_one("hsigmoid", 2, 0, 1024);
    run_one("mul", 3, 0, 1024);
    run_one("relu", 4, 0, 1024);
    run_one("add", 5, 0, 1024);
    run_one("sum", 6, 0, 1024);
    run_one("mean", 7, 0, 4);
    run_one("requant_stream", 0, 1, 512);
    run_one("hswish_stream", 1, 1, 512);
    run_one("hsigmoid_stream", 2, 1, 512);
    phase.drop_objection(this);
  endtask
endclass
