// LAYERNORM latency: 1 row of 16 elements (4 bank rows), the fewest
// rows and elements LAYERNORM takes: its fixed cycles; LAYERNORM.odd takes 17
// elements, the fewest whose 1 / W takes the full division.
// One instruction per timing path: LayerNorm and RMSNorm, and the operands
// that change the latency of the dividers (emu/src/latency.rs): eps = 1.0,
// 1.5 and 2.0 with x = 0, so var + eps = eps.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); bbdev collects the printed cycles.
class layernorm_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(layernorm_latency_seq)
  bit [63:0] rs2;
  int unsigned iter;

  function new(string name = "layernorm_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `LAYERNORM_FUNCT7;
    // Input, output and parameters at bank row 0.
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
    item.wr_read_bank = 3;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class layernorm_latency_test extends uvm_test;
  `uvm_component_utils(layernorm_latency_test)

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
        layernorm_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  // One instruction from reset: x, gamma and beta zero, eps (FP32 bits) at
  // its bank row after gamma (and beta).
  task measure(string label, int unsigned iter, int unsigned w, bit rms, bit [31:0] eps);
    layernorm_latency_seq seq;
    int unsigned lines = (w + 3) / 4;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
    end
    env.mem_model.load_word_g(2, 0, rms ? lines : 2 * lines, 128'(eps));
    env.mem_model.arm();
    seq = layernorm_latency_seq::type_id::create("seq");
    seq.rs2 = 64'(w) | (64'(rms) << 58);
    seq.iter = iter;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    measure("LAYERNORM", 1, 16, 0, 32'h3f800000);
    measure("LAYERNORM.even", 1, 16, 0, 32'h3fc00000);
    measure("LAYERNORM.odd", 1, 17, 0, 32'h40000000);
    measure("LAYERNORM.rms", 1, 16, 1, 32'h3f800000);
    phase.drop_objection(this);
  endtask
endclass
