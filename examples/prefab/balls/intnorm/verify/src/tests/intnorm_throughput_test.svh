// INTNORM throughput: one 8-row instruction of K = 512, the most rows the
// 1024-row input bank holds (iter * K / 4 <= 1024): the rows per cycle of a
// full instruction.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once. Every
// input, gamma and beta is 0 and epsilon is 1; all bases are 0.
// The shared cycle checker holds it, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs). Prints the cycles as the result.
class intnorm_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(intnorm_throughput_seq)

  function new(string name = "intnorm_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = `INTNORM_FUNCT7;
    item.rs2          = 512;
    item.iter         = 8;
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

class intnorm_throughput_test extends uvm_test;
  `uvm_component_utils(intnorm_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(intnorm_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    intnorm_throughput_seq seq;
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 1024; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, row == 0 ? 128'd1 : '0);
    end
    env.mem_model.arm();
    seq = intnorm_throughput_seq::type_id::create("seq");
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF INTNORM %0d %0d", 8,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass
