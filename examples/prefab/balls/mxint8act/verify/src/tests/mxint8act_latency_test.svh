// MXINT8ACT latency: one row of K=32, the smallest legal shape (one block): the fixed cycles of each path.
// One instruction per timing path, issued one at a time, with data that sends
// every lane down that path:
//   tail      codes 0x7f, exponent 0 (scale 0x7f): x = 127 << 26, high tail
//   interior  codes 0, exponent 0: table entry 2048, word 0 of row 512
//   next      codes 3, exponent -7 (scale 0x78): x = 3 << 19, table entry
//             2051, word 3 of row 512, so its neighbour is read from row 513
//   gated     the interior data, gate codes 0 and gate exponent 0
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// Prints the cycles from the command handshake to the response handshake;
// the shared cycle checker compares every instruction with BEMU's cycle
// model (emu/src/latency.rs).
class mxint8act_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxint8act_latency_seq)
  int unsigned iter, k, gated;

  function new(string name = "mxint8act_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `MXINT8ACT_FUNCT7;
    item.rs2          = 64'(gated) | (64'(k) << 1);
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
    item.wr_read_bank = 3;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class mxint8act_latency_test extends uvm_test;
  `uvm_component_utils(mxint8act_latency_test)

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
        mxint8act_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    mxint8act_cycles model;
    super.connect_phase(phase);
    $cast(model, env.cycles);
    model.mem_model = env.mem_model;
  endfunction

  // Input rows of the row stride K/16 + ceil(K/512): K/16 code rows, then one
  // scale row (K <= 512). The gate half starts at a multiple of the stride, so
  // the same pattern covers it.
  task measure(string path, int unsigned iter, int unsigned k, int unsigned gated, bit [7:0] code,
               bit [7:0] scale);
    mxint8act_latency_seq seq;
    int unsigned stride = k / 16 + 1;
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++)
      env.mem_model.load_word_g(1, 0, row, row % stride < stride - 1 ? {16{code}} : {16{scale}});
    for (int unsigned row = 0; row < 1024; row++) env.mem_model.load_word_g(2, 0, row, '0);
    env.mem_model.arm();
    seq       = mxint8act_latency_seq::type_id::create("seq");
    seq.iter  = iter;
    seq.k     = k;
    seq.gated = gated;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF MXINT8ACT.%s %0d %0d", path, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    measure("tail", 1, 32, 0, 8'h7f, 8'h7f);
    measure("interior", 1, 32, 0, 8'h00, 8'h7f);
    measure("next", 1, 32, 0, 8'h03, 8'h78);
    measure("gated", 1, 32, 1, 8'h00, 8'h7f);
    phase.drop_objection(this);
  endtask
endclass
