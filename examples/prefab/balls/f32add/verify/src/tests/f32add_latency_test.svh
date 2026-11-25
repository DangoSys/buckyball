// F32ADD latency: one 1-row instruction per mode, the fewest rows F32ADD
// takes (the RTL asserts iter != 0): its fixed cycles.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// Both rs2[5] modes, each measured alone: F32ADD.acc adds source group 0 of
// bank 1 to bank 2 (both ports read), F32ADD.first adds it to +0 (port 1
// idle); the result goes to bank 3.
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); the BBPERF line reports the cycles.
class f32add_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(f32add_latency_seq)

  bit [63:0] rs2;

  function new(string name = "f32add_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = F32ADD_THREAD_FUNCT7[6:0];
    item.rs2          = rs2;
    item.iter         = 1;
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

class f32add_latency_test extends uvm_test;
  `uvm_component_utils(f32add_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(f32add_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
    end
    env.mem_model.arm();
    measure(64'h0, "F32ADD.acc");
    measure(64'h20, "F32ADD.first");
    phase.drop_objection(this);
  endtask

  task measure(bit [63:0] rs2, string label);
    f32add_latency_seq seq;
    seq = f32add_latency_seq::type_id::create("seq");
    seq.rs2 = rs2;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, 1,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
