// SMATMUL throughput: the largest workload of each path: K = 1024 with M = 16 (a whole A and
// B bank) or M = 1; 11 reduction groups (11 A + 11 B + C + bias = 24 banks);
// M = 256, K = 64 with C base 63 across two C groups.
// One instruction per path, each labelled with the path it takes:
//   SMATMUL_BIAS          the four bias rows
//   SMATMUL_OS.first      M a multiple of 16, first set: bias fill, no C write
//   SMATMUL_OS.middle     neither first nor last: accumulate only
//   SMATMUL_OS.last       last set: accumulate, then write C back
//   SMATMUL_OS.whole      first and last: bias fill, accumulate, write C back
//   SMATMUL_OS.vec.*      the same four for M = 1 (one A row per tile)
//   SMATMUL_OS.kgroups    several reduction groups (op1_col > 1)
//   SMATMUL_OS.cgroups    C crossing into a second group (wr_col = 2)
// Against the ideal outside world of the shared bank model: every request
// ready, a read answered the next cycle, the response accepted at once. The
// cycle checker compares each instruction with the BEMU cycle model; this test
// prints the cycles from the command handshake to the response handshake.
class smatmul_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(smatmul_throughput_seq)

  bit bias;
  int unsigned m;
  int unsigned k;
  bit first;
  bit last;
  int unsigned k_groups;
  int unsigned wr_col;
  int unsigned c_base;

  function new(string name = "smatmul_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.wr_read_bank = 0;
    if (bias) begin
      item.funct        = SMATMUL_BIAS_THREAD_FUNCT7[6:0];
      item.rs2          = 0;
      item.iter         = 4;
      item.op1_bank     = 4;
      item.op2_bank     = 0;
      item.wr_bank      = 0;
      item.op1_col      = 1;
      item.op2_col      = 0;
      item.wr_col       = 0;
      item.wr_arch_bank = 0;
    end else begin
      // rs2: M [11:0], N = 16 [23:12], first [24], last [25], C base [31:26].
      item.funct = SMATMUL_OS_THREAD_FUNCT7[6:0];
      item.rs2 = (64'(c_base) << 26) | (64'(last) << 25) | (64'(first) << 24) |
          (64'd16 << 12) | 64'(m);
      item.iter = k;
      item.op1_bank = 1;
      item.op2_bank = 2;
      item.wr_bank = 3;
      item.op1_col = k_groups;
      item.op2_col = k_groups;
      item.wr_col = wr_col;
      item.wr_arch_bank = 3;
    end
    finish_item(item);
  endtask
endclass

class smatmul_throughput_test extends uvm_test;
  `uvm_component_utils(smatmul_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(smatmul_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task measure(string label, bit bias, int unsigned m, int unsigned k, bit first, bit last,
               int unsigned k_groups, int unsigned wr_col, int unsigned c_base);
    smatmul_throughput_seq seq;
    seq = smatmul_throughput_seq::type_id::create("seq");
    seq.bias = bias;
    seq.m = m;
    seq.k = k;
    seq.first = first;
    seq.last = last;
    seq.k_groups = k_groups;
    seq.wr_col = wr_col;
    seq.c_base = c_base;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, bias ? 4 : k,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 4; row++) env.mem_model.load_word_g(4, 0, row, '0);
    for (int unsigned group = 0; group < 11; group++)
      for (int unsigned row = 0; row < 1024; row++) begin
        env.mem_model.load_word_g(1, group, row, '0);
        env.mem_model.load_word_g(2, group, row, '0);
      end
    env.mem_model.arm();
    measure("SMATMUL_BIAS", 1, 0, 0, 0, 0, 0, 0, 0);
    measure("SMATMUL_OS.first", 0, 16, 1024, 1, 0, 1, 1, 0);
    measure("SMATMUL_OS.middle", 0, 16, 1024, 0, 0, 1, 1, 0);
    measure("SMATMUL_OS.last", 0, 16, 1024, 0, 1, 1, 1, 0);
    measure("SMATMUL_OS.whole", 0, 16, 1024, 1, 1, 1, 1, 0);
    measure("SMATMUL_OS.vec.first", 0, 1, 1024, 1, 0, 1, 1, 0);
    measure("SMATMUL_OS.vec.middle", 0, 1, 1024, 0, 0, 1, 1, 0);
    measure("SMATMUL_OS.vec.last", 0, 1, 1024, 0, 1, 1, 1, 0);
    measure("SMATMUL_OS.vec.whole", 0, 1, 1024, 1, 1, 1, 1, 0);
    measure("SMATMUL_OS.kgroups", 0, 16, 1024, 1, 1, 11, 1, 0);
    measure("SMATMUL_OS.cgroups", 0, 256, 64, 1, 1, 1, 2, 63);
    phase.drop_objection(this);
  endtask
endclass
