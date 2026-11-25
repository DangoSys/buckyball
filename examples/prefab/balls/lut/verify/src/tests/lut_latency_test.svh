// LUT latency: one 1-row instruction per mode, the fewest rows
// LUT takes: its fixed cycles.
// rs2 = mode | multiplier << 32: byte lookup in a shared 16-row table (mode
// 0), word lookup (mode 1) and softmax (mode 2, multiplier Q16 one). Every
// table lane holds Q30 one, the entry softmax requires at index 0.
// Each instruction is issued alone after a reset, against the ideal outside
// world of the shared bank model: every request ready, a read answered the
// next cycle, the response accepted at once. Prints the cycles from the
// command handshake to the response handshake; the shared cycle checker
// holds it, like every instruction, to the Ball's BEMU cycle model
// (emu/src/latency.rs).
class lut_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(lut_latency_seq)

  longint unsigned rs2;
  int unsigned iter;

  function new(string name = "lut_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = LUT_THREAD_FUNCT7[6:0];
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

class lut_latency_test extends uvm_test;
  `uvm_component_utils(lut_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(lut_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("LUT.byte", 0, 1);
    measure("LUT.word", 1, 1);
    measure("LUT.softmax", 2 | (64'd1 << 48), 1);
    phase.drop_objection(this);
  endtask

  task measure(string label, longint unsigned rs2, int unsigned iter);
    lut_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < iter; row++) env.mem_model.load_word_g(1, 0, row, '0);
    for (int unsigned row = 0; row < LUT_LANE_TABLE_ROWS; row++)
      env.mem_model.load_word_g(2, 0, row, {4{32'h4000_0000}});
    env.mem_model.arm();
    seq = lut_latency_seq::type_id::create("seq");
    seq.rs2 = rs2;
    seq.iter = iter;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
