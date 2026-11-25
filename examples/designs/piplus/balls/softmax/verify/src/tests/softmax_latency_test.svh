// SOFTMAX latency: 1 row of 16 elements (4 bank rows), the fewest
// rows and elements SOFTMAX takes (20 for SOFTMAX.odd): its fixed cycles.
// x is zero, so every e is 1.0 and the sum is W: the latency of the divider
// depends on it (emu/src/latency.rs), one label per path: SOFTMAX a power of two,
// SOFTMAX.odd not.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); bbdev collects the printed cycles.
class softmax_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(softmax_latency_seq)
  int unsigned w;
  int unsigned iter;

  function new(string name = "softmax_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `SOFTMAX_FUNCT7;
    // Input and output at bank row 0, scale 2^0.
    item.rs2          = 64'(w);
    item.iter         = iter;
    item.op1_bank     = 1;
    item.op2_bank     = 0;
    item.wr_bank      = 2;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = 1;
    item.op2_col      = 0;
    item.wr_col       = 1;
    item.wr_read_bank = 2;
    item.wr_arch_bank = 2;
    finish_item(item);
  endtask
endclass

class softmax_latency_test extends uvm_test;
  `uvm_component_utils(softmax_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(softmax_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  // One instruction from reset on x = 0.
  task measure(string label, int unsigned iter, int unsigned w);
    softmax_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) env.mem_model.load_word_g(1, 0, row, '0);
    env.mem_model.arm();
    seq = softmax_latency_seq::type_id::create("seq");
    seq.w    = w;
    seq.iter = iter;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("SOFTMAX", 1, 16);
    measure("SOFTMAX.odd", 1, 20);
    phase.drop_objection(this);
  endtask
endclass
