// MXINT8MM latency: M = 1, N = 4, K = 32, the fewest MXINT8MM
// takes, two tiles (N = 12) for MXINT8MM.tiles: its fixed cycles.
// One instruction per timing path: one tile overwriting C (MXINT8MM),
// accumulating into the old C (MXINT8MM.acc), with bias (MXINT8MM.bias), and
// tiles whose stores hold the next tile back (MXINT8MM.tiles).
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); bbdev collects the printed cycles.
class mxint8mm_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxint8mm_latency_seq)
  bit [63:0] rs2;
  int unsigned iter;

  function new(string name = "mxint8mm_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `MXINT8MM_FUNCT7;
    // A in bank 1, B in bank 2, C in bank 3, the old C and the bias in bank 4.
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
    item.wr_read_bank = 4;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class mxint8mm_latency_test extends uvm_test;
  `uvm_component_utils(mxint8mm_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
        mxint8mm_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  // One instruction from reset on zero codes, exponents, old C and bias: A at
  // bank row 0, C at c_base, c_pitch bank rows a C row.
  task measure(string label, int unsigned m, int unsigned n, int unsigned k, int unsigned c_pitch,
               int unsigned c_base, bit accumulate, bit bias);
    mxint8mm_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
      env.mem_model.load_word_g(4, 0, row, '0);
    end
    env.mem_model.arm();
    seq = mxint8mm_latency_seq::type_id::create("seq");
    seq.rs2 = 64'(m) | (64'(n) << 10) | (64'(c_pitch) << 34) | (64'(c_base) << 48) |
        (64'(accumulate) << 62) | (64'(bias) << 63);
    seq.iter = k;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, k,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("MXINT8MM", 1, 4, 32, 2, 2, 0, 0);
    measure("MXINT8MM.acc", 1, 4, 32, 2, 2, 1, 0);
    measure("MXINT8MM.bias", 1, 4, 32, 2, 2, 0, 1);
    measure("MXINT8MM.tiles", 1, 12, 32, 3, 0, 0, 0);
    phase.drop_objection(this);
  endtask
endclass
