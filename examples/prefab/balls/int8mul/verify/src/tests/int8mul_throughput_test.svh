// INT8MUL throughput: 1024 rows per group, the whole bank: the rows per cycle
// of a full instruction.
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once.
// Gate row 0 of bank 1, ratio 1.0, input rows of bank 2, output to bank 3.
// Two paths, each one instruction measured alone: INT8MUL.group1 (one group)
// and INT8MUL.group2 (two groups: the gate row is read again per group).
// The shared cycle checker holds each, like every instruction, to the Ball's
// BEMU cycle model (emu/src/latency.rs); the BBPERF line reports the cycles.
class int8mul_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(int8mul_throughput_seq)

  int unsigned groups;

  function new(string name = "int8mul_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = INT8MUL_THREAD_FUNCT7[6:0];
    item.rs2          = 64'h0000_0000_3f80_0000;
    item.iter         = 1024;
    item.op1_bank     = 1;
    item.op2_bank     = 2;
    item.wr_bank      = 3;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = groups;
    item.op2_col      = groups;
    item.wr_col       = groups;
    item.wr_read_bank = 0;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class int8mul_throughput_test extends uvm_test;
  `uvm_component_utils(int8mul_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(int8mul_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned group = 0; group < 2; group++) begin
      env.mem_model.load_word_g(1, group, 0, '0);
      for (int unsigned row = 0; row < 1024; row++) env.mem_model.load_word_g(2, group, row, '0);
    end
    env.mem_model.arm();
    measure(1, "INT8MUL.group1");
    measure(2, "INT8MUL.group2");
    phase.drop_objection(this);
  endtask

  task measure(int unsigned groups, string label);
    int8mul_throughput_seq seq;
    seq = int8mul_throughput_seq::type_id::create("seq");
    seq.groups = groups;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, 1024,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask
endclass
