// INT8ADD and INT8ADD_RELU latency: one 1-row instruction each, the fewest
// rows they take (the RTL asserts iter != 0): their fixed cycles.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// Both ratios 1.0; banks 1 and 2 into bank 3. The instructions run one after
// the other, each measured alone.
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); the BBPERF line reports the cycles.
class int8add_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(int8add_latency_seq)

  bit [6:0] funct;

  function new(string name = "int8add_latency_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = funct;
    item.rs2          = 64'h3f80_0000_3f80_0000;
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

class int8add_latency_test extends uvm_test;
  `uvm_component_utils(int8add_latency_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(int8add_cycles::get_type());
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
    measure(INT8ADD_THREAD_FUNCT7[6:0], "INT8ADD");
    measure(INT8ADD_RELU_THREAD_FUNCT7[6:0], "INT8ADD_RELU");
    phase.drop_objection(this);
  endtask

  task measure(bit [6:0] funct, string mnemonic);
    int8add_latency_seq seq;
    seq = int8add_latency_seq::type_id::create("seq");
    seq.funct = funct;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", mnemonic, 1,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
