// MXINT8MM throughput: one M = N = 64, K = 224 instruction, the most
// MACs the RTL asserts allow in 1024-row banks (C: 64 * 64 / 4 = 1024 rows;
// A and B: 64 rows of 15 packed rows = 960): 128 full 4 x 8 tiles of 7 blocks,
// the MACs per cycle of a full instruction.
// Paths, each one instruction after its own reset:
//   MXINT8MM      every tile starts from +0.0 (rs2 bit 62 clear)
//   MXINT8MM.acc  rs2 bit 62: every tile first reads its old C rows (bank 4)
// A, B and C from row 0. Zero operands keep every FP32 term finite; the PE
// has no data-dependent timing.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, a write acknowledged the next cycle,
// the response accepted at once.
// Prints the cycles from the command handshake to the response handshake;
// bbdev compares them with perf.toml.
class mxint8matmul_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxint8matmul_throughput_seq)

  bit from_old;

  function new(string name = "mxint8matmul_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `MXINT8MM_FUNCT7;
    item.rs2          = (64'(from_old) << 62) | (64'(64) << 10) | 64'(64);
    item.iter         = 224;
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

class mxint8matmul_throughput_test extends uvm_test;
  `uvm_component_utils(mxint8matmul_throughput_test)

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
        mxint8matmul_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task measure(string label, bit from_old);
    mxint8matmul_throughput_seq seq;
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
    seq = mxint8matmul_throughput_seq::type_id::create("seq");
    seq.from_old = from_old;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, 224,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("MXINT8MM", 0);
    measure("MXINT8MM.acc", 1);
    phase.drop_objection(this);
  endtask
endclass
