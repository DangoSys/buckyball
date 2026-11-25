// MXQUANT throughput: one 4096-element instruction, the 4096 FP32 inputs of the whole 1024-row bank: the elements per cycle of a full instruction.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, a write acknowledged the next cycle,
// the response accepted at once.
// Prints the cycles from the command handshake to the response handshake;
// bbdev compares them with perf.toml.
class mxquant_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(mxquant_throughput_seq)

  function new(string name = "mxquant_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = MXQUANT_THREAD_FUNCT7[6:0];
    item.rs2          = 0;
    item.iter         = 4096;
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

class mxquant_throughput_test extends uvm_test;
  `uvm_component_utils(mxquant_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(mxquant_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    mxquant_throughput_seq seq;
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) env.mem_model.load_word_g(1, 0, row, '0);
    env.mem_model.arm();
    seq = mxquant_throughput_seq::type_id::create("seq");
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF MXQUANT %0d %0d", 4096,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass
