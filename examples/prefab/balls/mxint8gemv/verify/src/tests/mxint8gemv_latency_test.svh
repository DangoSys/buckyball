// MXINT8GEMV latency: K = 32, one block, the smallest legal K: the fixed
// cycles.
// Paths, each one instruction after its own reset:
//   MXINT8GEMV            plain: no old C read, no quantized output
//   MXINT8GEMV.acc        rs2 bit 56: first reads the 17-row old C state
//   MXINT8GEMV.quant      rs2 bit 57: encodes 32 codes, writes 3 rows
//   MXINT8GEMV.acc.quant  both
//   MXINT8GEMV.zero       plain with A all zero: every product is zero, so
//                         each block goes selectScale -> advance
// A codes from row 0, A exponents from row 512, B, new C and old C from row 0.
// Operand bytes are 0x7f (A zero for .zero): finite exponents and nonzero
// products, so every other block takes selectScale -> align -> add ->
// normalize. The old C (bank 4) is an empty state: header count 1, valid 0,
// scale 0; zero values.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, a write acknowledged the next cycle,
// the response accepted at once.
// Prints the cycles from the command handshake to the response handshake of
// each path; bbdev compares them with perf.toml.
class mxint8gemv_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxint8gemv_latency_seq)

  bit [63:0] rs2;

  function new(string name = "mxint8gemv_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `MXINT8GEMV_FUNCT7;
    item.rs2          = rs2 | (64'(512) << 14);
    item.iter         = 32;
    item.op1_bank     = 1;
    item.op2_bank     = 2;
    item.wr_bank      = 3;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = 1;
    item.op2_col      = 4;
    item.wr_col       = 1;
    item.wr_read_bank = 4;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class mxint8gemv_latency_test extends uvm_test;
  `uvm_component_utils(mxint8gemv_latency_test)

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
        mxint8gemv_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task measure(string label, bit [63:0] rs2, bit [7:0] a_byte);
    mxint8gemv_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, {16{a_byte}});
      for (int unsigned g = 0; g < 4; g++) env.mem_model.load_word_g(2, g, row, {16{8'h7f}});
      env.mem_model.load_word_g(4, 0, row, row == 0 ? 128'(1) : '0);
    end
    env.mem_model.arm();
    seq = mxint8gemv_latency_seq::type_id::create("seq");
    seq.rs2 = rs2;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, 32,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("MXINT8GEMV", 0, 8'h7f);
    measure("MXINT8GEMV.acc", 64'(1) << 56, 8'h7f);
    measure("MXINT8GEMV.quant", 64'(1) << 57, 8'h7f);
    measure("MXINT8GEMV.acc.quant", 64'(3) << 56, 8'h7f);
    measure("MXINT8GEMV.zero", 0, 8'h00);
    phase.drop_objection(this);
  endtask
endclass
