// REDMULE_F16 throughput: every timing path with its largest legal workload
// in the 1024-row default bank.
// Each path runs alone after a reset, against the ideal outside world of the
// shared bank model: every request ready, a read answered the next cycle, a
// write acknowledged the next cycle, the response accepted at once. Prints the
// cycles from the command handshake to the response handshake; the shared
// cycle checker holds every instruction to the BEMU cycle model
// (emu/src/latency.rs).
class redmule_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(redmule_throughput_seq)
  int rows,  k;
  bit first;

  function new(string name = "redmule_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    // bb_redmule(a=0, b=1, c=2, m, k, bases 0, first); a continuation reads
    // the previous C from bank 3, its wr_read_bank.
    item.rob_id       = 0;
    item.funct        = `REDMULE_F16_FUNCT7;
    item.rs2          = longint'(rows) | (longint'(first) << 42);
    item.iter         = k;
    item.op1_bank     = 0;
    item.op2_bank     = 1;
    item.wr_bank      = 2;
    item.op1_group    = 0;
    item.op2_group    = 0;
    item.wr_group     = 0;
    item.op1_col      = 1;
    item.op2_col      = 1;
    item.wr_col       = 1;
    item.wr_read_bank = first ? 0 : 3;
    item.wr_arch_bank = 2;
    finish_item(item);
  endtask
endclass

class redmule_throughput_test extends uvm_test;
  `uvm_component_utils(redmule_throughput_test)

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
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(redmule_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction


  // One instruction of M = rows, K = k: preload the rows it reads (A: M * K / 8
  // rows of bank 0, B: K rows of bank 1, C: M rows of bank 3 when not first).
  task run_path(string label, int rows, int k, bit first);
    redmule_throughput_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int row = 0; row < rows * k / 8; row++) env.mem_model.load_word_g(0, 0, row, '0);
    for (int row = 0; row < k; row++) env.mem_model.load_word_g(1, 0, row, '0);
    if (!first) for (int row = 0; row < rows; row++) env.mem_model.load_word_g(3, 0, row, '0);
    env.mem_model.arm();
    seq       = redmule_throughput_seq::type_id::create("seq");
    seq.rows  = rows;
    seq.k     = k;
    seq.first = first;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", label, k,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    // One tile (M = 8), K = 1024: the whole B bank and, with it, M * K / 8 =
    // 1024 A rows; C written only.
    run_path("REDMULE_F16", 8, 1024, 1);
    // The same with the previous C read (first = 0, GEMM).
    run_path("REDMULE_F16.ACC", 8, 1024, 0);
    // The most row tiles: M = 1024 (the whole C bank), K = 8.
    run_path("REDMULE_F16.TILES", 1024, 8, 1);
    run_path("REDMULE_F16.ACC_TILES", 1024, 8, 0);
    phase.drop_objection(this);
  endtask
endclass
