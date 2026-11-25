// MXMM throughput: every timing path once with the largest matmul the
// 1024-row default bank holds, M = N = 64, K = 224 (the windowed A with
// startK != 0 needs fullK > K: K = 192, fullK = 224), the MACs per cycle of a
// full instruction.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// bb_blink_env checks each instruction against the cycle model
// (casegen/src/latency.rs); the BBPERF lines are bbdev's results.
class mxmm_perf_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxmm_perf_seq)

  bit [6:0] funct;
  bit [63:0] rs2;
  int unsigned k;

  function new(string name = "mxmm_perf_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = funct;
    item.rs2          = rs2;
    item.iter         = k;
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

class mxmm_throughput_test extends uvm_test;
  `uvm_component_utils(mxmm_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(mxmm_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    // rs2 = {startK, fullK, base 0, last, first, N, M}.
    restart();
    issue("MXMM_MXFP8.first_last", `MXMM_MXFP8_FUNCT7, {32'd0, 6'd0, 2'b11, 12'd64, 12'd64}, 224);
    // One chain: first (no output), a continuation (reads the accumulators,
    // no output), last (reads the accumulators, writes C).
    restart();
    issue("MXMM_MXFP8.first", `MXMM_MXFP8_FUNCT7, {32'd0, 6'd0, 2'b01, 12'd64, 12'd64}, 224);
    issue("MXMM_MXFP8.mid", `MXMM_MXFP8_FUNCT7, {32'd0, 6'd0, 2'b00, 12'd64, 12'd64}, 224);
    issue("MXMM_MXFP8.last", `MXMM_MXFP8_FUNCT7, {32'd0, 6'd0, 2'b10, 12'd64, 12'd64}, 224);
    // The window over the whole A, then over part of a wider A.
    restart();
    issue("MXMM_MXFP8_WINDOW.first_last", `MXMM_MXFP8_WINDOW_FUNCT7, {
          16'd0, 16'd224, 6'd0, 2'b11, 12'd64, 12'd64}, 224);
    restart();
    issue("MXMM_MXFP8_WINDOW.offset", `MXMM_MXFP8_WINDOW_FUNCT7, {
          16'd32, 16'd224, 6'd0, 2'b11, 12'd64, 12'd64}, 192);
    phase.drop_objection(this);
  endtask

  // Reset (no live chain) and preload every row of A (bank 1) and B (bank 2).
  task restart();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
    end
    env.mem_model.arm();
  endtask

  task issue(string label, int funct, bit [63:0] rs2, int unsigned k);
    mxmm_perf_seq seq;
    seq = mxmm_perf_seq::type_id::create("seq");
    seq.funct = funct[6:0];
    seq.rs2 = rs2;
    seq.k = k;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, k,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
