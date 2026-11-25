// MAXPOOL and TOPK_U32 latency: the smallest instruction of each path: a
// 1x1 MAXPOOL, a 1x1 MAXPOOL whose only position is padding, a TOPK_U32 of
// k = 1 over one row in one group and in two groups.
// MAXPOOL rs2 packs inSide, outSide, kernel, stride, padding (4 bits each
// from bit 0) and outStride at [37:32]; TOPK_U32 rs2 = n | k << 32. rows is
// the input rows read in each of the cols input groups.
// Each instruction is issued alone after a reset, against the ideal outside
// world of the shared bank model: every request ready, a read answered the
// next cycle, the response accepted at once. Prints the cycles from the
// command handshake to the response handshake; the shared cycle checker
// holds it, like every instruction, to the Ball's BEMU cycle model
// (emu/src/latency.rs).
class maxpool_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(maxpool_latency_seq)

  int unsigned funct;
  longint unsigned rs2;
  int unsigned iter;
  int unsigned cols;
  int unsigned wr_cols;

  function new(string name = "maxpool_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = funct[6:0];
    item.rs2          = rs2;
    item.iter         = iter;
    item.op1_bank     = 1;
    item.op2_bank     = 0;
    item.wr_bank      = 2;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = cols;
    item.op2_col      = 0;
    item.wr_col       = wr_cols;
    item.wr_read_bank = 0;
    item.wr_arch_bank = 2;
    finish_item(item);
  endtask
endclass

class maxpool_latency_test extends uvm_test;
  `uvm_component_utils(maxpool_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(maxpool_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("MAXPOOL", MAXPOOL_THREAD_FUNCT7,
            64'd1 | (64'd1 << 4) | (64'd1 << 8) | (64'd1 << 12) | (64'd0 << 16) | (64'd1 << 32), 1,
            1, 1, 1);
    measure("MAXPOOL.pad", MAXPOOL_THREAD_FUNCT7,
            64'd1 | (64'd1 << 4) | (64'd1 << 8) | (64'd1 << 12) | (64'd1 << 16) | (64'd1 << 32), 1,
            1, 1, 0);
    measure("TOPK_U32", TOPK_THREAD_FUNCT7, 64'd1 | (64'd1 << 32), 1, 1, 1, 1);
    measure("TOPK_U32.groups", TOPK_THREAD_FUNCT7, 64'd1 | (64'd1 << 32), 1, 2, 1, 1);
    phase.drop_objection(this);
  endtask

  task measure(string label, int unsigned funct, longint unsigned rs2, int unsigned iter,
               int unsigned cols, int unsigned wr_cols, int unsigned rows);
    maxpool_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned group = 0; group < cols; group++)
      for (int unsigned row = 0; row < rows; row++) env.mem_model.load_word_g(1, group, row, '0);
    env.mem_model.arm();
    seq = maxpool_latency_seq::type_id::create("seq");
    seq.funct = funct;
    seq.rs2 = rs2;
    seq.iter = iter;
    seq.cols = cols;
    seq.wr_cols = wr_cols;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
