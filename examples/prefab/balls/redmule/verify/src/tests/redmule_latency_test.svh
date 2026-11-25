// REDMULE_F16 latency: every timing path with its smallest legal workload
// (M >= 1, K a nonzero multiple of 8).
// Each path runs alone after a reset, against the ideal outside world of the
// shared bank model: every request ready, a read answered the next cycle, a
// write acknowledged the next cycle, the response accepted at once. Prints the
// cycles from the command handshake to the response handshake; the shared
// cycle checker holds every instruction to the BEMU cycle model
// (emu/src/latency.rs).
class redmule_latency_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(redmule_latency_seq)
  int rows,  k;
  bit first;

  function new(string name = "redmule_latency_seq");
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

class redmule_latency_test extends uvm_test;
  `uvm_component_utils(redmule_latency_test)

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
    redmule_latency_seq seq;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int row = 0; row < rows * k / 8; row++) env.mem_model.load_word_g(0, 0, row, '0);
    for (int row = 0; row < k; row++) env.mem_model.load_word_g(1, 0, row, '0);
    if (!first) for (int row = 0; row < rows; row++) env.mem_model.load_word_g(3, 0, row, '0);
    env.mem_model.arm();
    seq       = redmule_latency_seq::type_id::create("seq");
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
    // The smallest shape: M = 1, K = 8 (one tile, K promoted to 16 W rows).
    run_path("REDMULE_F16", 1, 8, 1);
    run_path("REDMULE_F16.ACC", 1, 8, 0);
    // The fewest rows that take two row tiles: M = 9, K = 8.
    run_path("REDMULE_F16.TILES", 9, 8, 1);
    run_path("REDMULE_F16.ACC_TILES", 9, 8, 0);
    phase.drop_objection(this);
  endtask
endclass
