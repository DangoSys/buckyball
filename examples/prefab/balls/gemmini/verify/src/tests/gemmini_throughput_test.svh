// Gemmini throughput: every instruction in each mode its cycles depend on,
// each with its largest workload: PRELOAD and the two COMPUTEs on the whole
// 16x16 tile, the loops on large matmul and convolution layers; the
// configurations have one workload.
// Each instruction runs alone after a reset, against the ideal outside world:
// the shared bank model for BLink, and for the SubROB expansion port a
// metadata read answered the next cycle and every write, row and completion
// accepted at once. gemmini_cycles and gemmini_expansion_cycles check every
// instruction against the BEMU cycle model (emu/src/latency.rs); this prints
// the cycles from the command handshake to the response handshake (BLink),
// the completion handshake (a loop configuration) or the last row handshake
// (a loop) as `BBPERF <label> <iter> <cycles>`.
class gemmini_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(gemmini_throughput_seq)

  bit [ 6:0] funct;
  bit [63:0] rs2;
  int        iter;

  function new(string name = "gemmini_throughput_seq");
    super.new(name);
  endfunction

  task body();
    bb_blink_cmd_item item;
    item = bb_blink_cmd_item::type_id::create("item");
    start_item(item);
    item.rob_id       = 0;
    item.funct        = funct;
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
    item.wr_col       = 4;
    item.wr_read_bank = 3;
    item.wr_arch_bank = 3;
    finish_item(item);
  endtask
endclass

class gemmini_throughput_test extends uvm_test;
  `uvm_component_utils(gemmini_throughput_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  virtual gemmini_expansion_if xif;
  bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;
  gemmini_expansion_cycles expansion_cycles;

  // The configurations of the measured modes (GEMMINI_CONFIG rs2).
  localparam bit [63:0] Os = 64'd0;
  localparam bit [63:0] Ws = 64'd1 << 4;
  localparam bit [63:0] At = 64'd1 << 7;
  localparam bit [63:0] Bt = 64'd1 << 8;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    if (!uvm_config_db#(virtual gemmini_expansion_if)::get(this, "", "expansion", xif))
      `uvm_fatal("NOVIF", "gemmini_expansion_if not found")
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(gemmini_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
    expansion_cycles = gemmini_expansion_cycles::type_id::create("expansion_cycles", this);
  endfunction

  task run_phase(uvm_phase phase);
    bit [63:0] ws_bounds;
    phase.raise_objection(this);
    xif.command_valid  = 1'b0;
    xif.complete_ready = 1'b1;
    xif.write_ready    = 1'b1;
    // Each after a GEMMINI_CONFIG of its mode: rs2[4] WS, rs2[7] a_transpose,
    // rs2[8] b_transpose (the paths of emu/src/latency.rs).
    blink("GEMMINI_CONFIG", `GEMMINI_CONFIG_FUNCT7, 64'd0, 0, Os, 0);
    blink("GEMMINI_FLUSH", `GEMMINI_FLUSH_FUNCT7, 64'd4, 0, Os, 0);
    blink("GEMMINI_PRELOAD.os", `GEMMINI_PRELOAD_FUNCT7, 64'd1, 16, Os, 0);
    blink("GEMMINI_PRELOAD.ws", `GEMMINI_PRELOAD_FUNCT7, 64'd1, 16, Ws, 0);
    blink("GEMMINI_PRELOAD.ws_bt", `GEMMINI_PRELOAD_FUNCT7, 64'd1, 16, Ws | Bt, 0);
    blink("GEMMINI_COMPUTE_PRELOADED.os", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2, 16, Os, 1);
    blink("GEMMINI_COMPUTE_PRELOADED.os_at", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2, 16, At, 1);
    blink("GEMMINI_COMPUTE_PRELOADED.os_at_bt", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2, 16,
          At | Bt, 1);
    blink("GEMMINI_COMPUTE_PRELOADED.ws", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2, 16, Ws, 1);
    blink("GEMMINI_COMPUTE_PRELOADED.ws_at_bt", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2, 16,
          Ws | At | Bt, 1);
    // zero_op2 (rs2[4]) and zero_op1_tail (rs2[5]), as LOOP_CONV_WS issues them.
    blink("GEMMINI_COMPUTE_PRELOADED.ws_zero", `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'h32, 16, Ws,
          1);
    blink("GEMMINI_COMPUTE_ACCUMULATED.os", `GEMMINI_COMPUTE_ACCUMULATED_FUNCT7, 64'd3, 16, Os, 0);
    blink("GEMMINI_COMPUTE_ACCUMULATED.os_at", `GEMMINI_COMPUTE_ACCUMULATED_FUNCT7, 64'd3, 16, At,
          0);
    expand("GEMMINI_LOOP_WS_CONFIG_BOUNDS", `GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_ADDR_A", `GEMMINI_LOOP_WS_CONFIG_ADDR_A_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_ADDR_B", `GEMMINI_LOOP_WS_CONFIG_ADDR_B_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_ADDR_D", `GEMMINI_LOOP_WS_CONFIG_ADDR_D_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_ADDR_C", `GEMMINI_LOOP_WS_CONFIG_ADDR_C_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_STRIDES_AB", `GEMMINI_LOOP_WS_CONFIG_STRIDES_AB_FUNCT7, 0);
    expand("GEMMINI_LOOP_WS_CONFIG_STRIDES_DC", `GEMMINI_LOOP_WS_CONFIG_STRIDES_DC_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_1", `GEMMINI_LOOP_CONV_WS_CONFIG_1_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_2", `GEMMINI_LOOP_CONV_WS_CONFIG_2_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_3", `GEMMINI_LOOP_CONV_WS_CONFIG_3_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_4", `GEMMINI_LOOP_CONV_WS_CONFIG_4_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_5", `GEMMINI_LOOP_CONV_WS_CONFIG_5_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_6", `GEMMINI_LOOP_CONV_WS_CONFIG_6_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_7", `GEMMINI_LOOP_CONV_WS_CONFIG_7_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_8", `GEMMINI_LOOP_CONV_WS_CONFIG_8_FUNCT7, 0);
    expand("GEMMINI_LOOP_CONV_WS_CONFIG_9", `GEMMINI_LOOP_CONV_WS_CONFIG_9_FUNCT7, 0);
    // LOOP_WS of a 256x256x256 matmul: max_i = max_j = max_k = 16 tiles.
    for (int i = 0; i < 136; i++) xif.metadata[i] = 8'd0;
    ws_bounds = (64'd16 << 32) | (64'd16 << 16) | 64'd16;
    for (int i = 0; i < 8; i++) xif.metadata[i] = ws_bounds[i*8+:8];
    expand("GEMMINI_LOOP_WS", `GEMMINI_LOOP_WS_FUNCT7, 16 * 16 * 16);
    // LOOP_CONV_WS of a 3x3 convolution, 64 to 64 channels, 18x18 input to
    // 16x16 output: 256 * 4 output tiles of 9 * 4 compute tiles each.
    for (int i = 0; i < 136; i++) xif.metadata[i] = 8'd0;
    conv_metadata(1, 18, 64, 64, 16, 3);
    expand("GEMMINI_LOOP_CONV_WS", `GEMMINI_LOOP_CONV_WS_FUNCT7, 256 * 4 * 9 * 4);
    phase.drop_objection(this);
  endtask

  // Conv metadata bytes 64..: batch, in_dim, in_channels | out_channels,
  // out_dim, stride 1, padding 0 | kernel_dim.
  function void conv_metadata(int batch, int in_dim, int in_channels, int out_channels, int out_dim,
                              int kernel_dim);
    bit [63:0] value0 = (64'(in_channels) << 32) | (64'(in_dim) << 16) | 64'(batch);
    bit [63:0] value1 = (64'd1 << 32) | (64'(out_dim) << 16) | 64'(out_channels);
    bit [63:0] value2 = 64'(kernel_dim);
    for (int i = 0; i < 8; i++) begin
      xif.metadata[64+i] = value0[i*8+:8];
      xif.metadata[72+i] = value1[i*8+:8];
      xif.metadata[80+i] = value2[i*8+:8];
    end
  endfunction

  task restart();
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    env.mem_model.clear_mem();
    for (int unsigned row = 0; row < 16; row++) begin
      env.mem_model.load_word_g(1, 0, row, '0);
      env.mem_model.load_word_g(2, 0, row, '0);
      for (int unsigned group = 0; group < 4; group++) env.mem_model.load_word_g(3, group, row, '0);
    end
    env.mem_model.arm();
  endtask

  task issue(bit [6:0] funct, bit [63:0] rs2, int iter);
    gemmini_throughput_seq seq;
    seq       = gemmini_throughput_seq::type_id::create("seq");
    seq.funct = funct;
    seq.rs2   = rs2;
    seq.iter  = iter;
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
  endtask

  // One BLink instruction after the GEMMINI_CONFIG of its mode; a
  // COMPUTE_PRELOADED computes on the tile a PRELOAD of its rows put in the
  // array, so that PRELOAD goes first, the compute right after its response
  // while the PRELOAD's rows still leave the mesh.
  task blink(string name, bit [6:0] funct, bit [63:0] rs2, int iter, bit [63:0] mode,
             bit preload_first);
    restart();
    issue(`GEMMINI_CONFIG_FUNCT7, mode, 0);
    if (preload_first) issue(`GEMMINI_PRELOAD_FUNCT7, 64'd1, iter);
    issue(funct, rs2, iter);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", name, iter,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
  endtask

  // One expansion command: banks A = 1, B = 2, C = 3 and low_d for the loops.
  task expand(string name, bit [6:0] funct, int iter);
    restart();
    @(negedge vif.clock);
    xif.command_bits_rob_id = 0;
    xif.command_bits_funct  = funct;
    xif.command_bits_rs2    = (64'd1 << 30) | (64'd3 << 20) | (64'd2 << 10) | 64'd1;
    xif.command_valid       = 1'b1;
    do @(posedge vif.clock); while (!xif.command_ready);
    @(negedge vif.clock);
    xif.command_valid = 1'b0;
    @(posedge vif.clock iff ((xif.complete_valid && xif.complete_ready) ||
                             (xif.rows_valid && xif.rows_bits_last)));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF %s %0d %0d", name, iter,
                                  xif.perf_completed - xif.perf_accepted), UVM_LOW)
  endtask
endclass
