class gemmini_ball_test extends uvm_test;
  `uvm_component_utils(gemmini_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  virtual gemmini_expansion_if xif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    if (!uvm_config_db#(virtual gemmini_expansion_if)::get(this, "", "expansion", xif))
      `uvm_fatal("NOVIF", "gemmini_expansion_if not found")
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    apply_reset();
    run_config();
    run_loop_bounds();
    run_group_select();
    phase.drop_objection(this);
  endtask

  // bb_gemmini_config(dataflow=1, activation=0, a_transpose=1, b_transpose=0, in_shift=3):
  // a BLink command answered by the execution controller without bank traffic.
  task run_config();
    int cycles;
    bit [63:0] rs2 = (64'd1 << 4) | (64'd1 << 7) | (64'd3 << 9);
    @(negedge vif.clock);
    vif.cmd_req_bits_rob_id       = 'h5;
    vif.cmd_req_bits_funct        = `GEMMINI_CONFIG_FUNCT7;
    vif.cmd_req_bits_rs2          = rs2;
    vif.cmd_req_bits_iter         = '0;
    vif.cmd_req_bits_op1_bank     = '0;
    vif.cmd_req_bits_op2_bank     = '0;
    vif.cmd_req_bits_wr_bank      = '0;
    vif.cmd_req_bits_op1_group    = '0;
    vif.cmd_req_bits_op2_group    = '0;
    vif.cmd_req_bits_wr_group     = '0;
    vif.cmd_req_bits_op1_col      = '0;
    vif.cmd_req_bits_op2_col      = '0;
    vif.cmd_req_bits_wr_col       = '0;
    vif.cmd_req_bits_wr_read_bank = '0;
    vif.cmd_req_bits_wr_arch_bank = '0;
    vif.cmd_req_valid             = 1'b1;
    cycles                        = 0;
    do begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > 20) `uvm_fatal("TIMEOUT", "Gemmini config command was not accepted")
    end while (!vif.cmd_req_ready);
    @(negedge vif.clock);
    vif.cmd_req_valid = 1'b0;

    cycles = 0;
    while (!vif.cmd_resp_valid) begin
      check_no_bank_traffic("config");
      @(negedge vif.clock);
      cycles++;
      if (cycles > 20) `uvm_fatal("RESP", "Gemmini config response was not observed")
    end
    verification_pkg::evidence::checks++;
    if (vif.cmd_resp_bits_rob_id !== 'h5)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0h", vif.cmd_resp_bits_rob_id))
    @(negedge vif.clock);
    `uvm_info("GEMMINI", "config command passed", UVM_LOW)
  endtask

  // bb_gemmini_loop_ws_config_bounds(max_i=1, max_j=2, max_k=3) arrives on the SubROB
  // expansion port and is stored little-endian into loop metadata bytes 0..7.
  task run_loop_bounds();
    int cycles;
    int unsigned writes;
    bit [63:0] rs2 = 64'h0001_0002_0003;
    @(negedge vif.clock);
    xif.command_bits_rob_id = 'h9;
    xif.command_bits_funct  = `GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNCT7;
    xif.command_bits_rs2    = rs2;
    xif.command_valid       = 1'b1;
    #1;
    verification_pkg::evidence::checks++;
    if (xif.accepts !== 1'b1 || xif.starts !== 1'b0)
      `uvm_fatal("EXPANSION", "loop bounds config must be accepted without starting a loop")
    cycles = 0;
    do begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > 20) `uvm_fatal("TIMEOUT", "loop bounds command was not accepted")
    end while (!xif.command_ready);
    @(negedge vif.clock);
    xif.command_valid = 1'b0;

    writes = 0;
    cycles = 0;
    while (!xif.complete_valid) begin
      check_no_bank_traffic("loop bounds");
      if (xif.write_valid) begin
        verification_pkg::evidence::checks++;
        if (xif.write_bits_addr !== writes || xif.write_bits_data !== rs2[writes*8+:8])
          `uvm_fatal("EXPANSION", $sformatf(
                     "metadata write %0d: addr=%0d data=%0h",
                     writes,
                     xif.write_bits_addr,
                     xif.write_bits_data
                     ))
        writes++;
      end
      @(negedge vif.clock);
      cycles++;
      if (cycles > 40) `uvm_fatal("TIMEOUT", "loop bounds command did not complete")
    end
    verification_pkg::evidence::checks++;
    if (writes != 8 || xif.complete_bits_rob_id !== 'h9)
      `uvm_fatal("EXPANSION", $sformatf(
                 "completion mismatch: writes=%0d rob=%0h", writes, xif.complete_bits_rob_id))
    @(negedge vif.clock);
    `uvm_info("GEMMINI", "loop-ws bounds config passed", UVM_LOW)
  endtask

  // Operands span three groups of their banks; rs2[41:36] and rs2[47:42] pick
  // A's group 2 and B's group 1. The other groups hold different data, so a
  // wrong group fails both the read check and the product.
  localparam int GroupA = 2;
  localparam int GroupB = 1;

  // Row `row` of group `group` of bank 1 (A) or bank 2 (B).
  function bit [127:0] bank_row(int bank, int group, int row);
    bit [127:0] data;
    for (int lane = 0; lane < 16; lane++)
    if (bank == 1) data[lane*8+:8] = group == GroupA ? 8'(lane == row % 16) : 8'd5;
    else data[lane*8+:8] = group == GroupB ? 8'((row + 2 * lane + 1) % 7 - 3) : 8'd9;
    return data;
  endfunction

  // Issues one command and serves its bank traffic until it completes:
  // reads answer one cycle after they fire, writes are recorded in c.
  task issue(int rob, int funct, bit [63:0] rs2, int op1_bank, int op2_bank,
             ref bit [31:0] c[16][16], ref int written);
    bit read_fire[`BB_IN_BW], write_fire[`BB_OUT_BW];
    bit read_done[`BB_IN_BW], write_done[`BB_OUT_BW];
    int read_bank[`BB_IN_BW], read_group[`BB_IN_BW], read_addr[`BB_IN_BW];
    bit cmd_fire = 0;
    int cycles;
    @(negedge vif.clock);
    vif.cmd_req_bits_rob_id       = rob;
    vif.cmd_req_bits_funct        = funct;
    vif.cmd_req_bits_rs2          = rs2;
    vif.cmd_req_bits_iter         = 16;
    vif.cmd_req_bits_op1_bank     = op1_bank;
    vif.cmd_req_bits_op2_bank     = op2_bank;
    vif.cmd_req_bits_wr_bank      = 3;
    vif.cmd_req_bits_op1_group    = '0;
    vif.cmd_req_bits_op2_group    = '0;
    vif.cmd_req_bits_wr_group     = '0;
    vif.cmd_req_bits_op1_col      = 3;
    vif.cmd_req_bits_op2_col      = 3;
    vif.cmd_req_bits_wr_col       = 4;
    vif.cmd_req_bits_wr_read_bank = '0;
    vif.cmd_req_bits_wr_arch_bank = 3;
    vif.cmd_req_valid             = 1'b1;
    cycles                        = 0;
    forever begin
      // Values set at this negedge are sampled with the DUT outputs at the
      // next posedge, so handshakes seen here fire at that edge.
      if (cmd_fire) vif.cmd_req_valid = 1'b0;
      for (int port = 0; port < `BB_IN_BW; port++) begin
        if (read_done[port]) vif.bank_read_resp_valid[port] = 1'b0;
        if (read_fire[port]) begin
          if (vif.bank_read_resp_valid[port]) `uvm_fatal("BANK", "read response overrun")
          vif.bank_read_resp_valid[port] = 1'b1;
          vif.bank_read_resp_data[port] =
              bank_row(read_bank[port], read_group[port], read_addr[port]);
        end
      end
      for (int port = 0; port < `BB_OUT_BW; port++) begin
        if (write_done[port]) vif.bank_write_resp_valid[port] = 1'b0;
        if (write_fire[port]) vif.bank_write_resp_valid[port] = 1'b1;
      end
      #1;
      cmd_fire = vif.cmd_req_valid && vif.cmd_req_ready;
      for (int port = 0; port < `BB_IN_BW; port++) begin
        read_done[port] = vif.bank_read_resp_valid[port] && vif.bank_read_resp_ready[port];
        read_fire[port] = vif.bank_read_req_valid[port] && vif.bank_read_req_ready[port];
        if (read_fire[port]) begin
          read_bank[port]  = vif.bank_read_bank_id[port];
          read_group[port] = vif.bank_read_group_id[port];
          read_addr[port]  = vif.bank_read_req_addr[port];
          verification_pkg::evidence::checks++;
          if (read_bank[port] != op1_bank && read_bank[port] != op2_bank)
            `uvm_fatal("BANK", $sformatf("read of bank %0d", read_bank[port]))
          if (read_group[port] != (read_bank[port] == 1 ? GroupA : GroupB))
            `uvm_fatal("GROUP", $sformatf(
                       "bank %0d read group %0d", read_bank[port], read_group[port]))
        end
      end
      for (int port = 0; port < `BB_OUT_BW; port++) begin
        write_done[port] = vif.bank_write_resp_valid[port] && vif.bank_write_resp_ready[port];
        write_fire[port] = vif.bank_write_req_valid[port] && vif.bank_write_req_ready[port];
        if (write_fire[port]) begin
          int group = vif.bank_write_group_id[port];
          int row = vif.bank_write_req_addr[port];
          if (vif.bank_write_bank_id[port] != 3 || group > 3 || row > 15 ||
              vif.bank_write_req_mask[port] !== 16'hffff)
            `uvm_fatal("BANK", $sformatf(
                       "write bank %0d group %0d row %0d", vif.bank_write_bank_id[port], group, row
                       ))
          for (int lane = 0; lane < 4; lane++)
          c[row][group*4+lane] = vif.bank_write_req_data[port][lane*32+:32];
          written++;
        end
      end
      if (vif.cmd_resp_valid && vif.cmd_resp_ready) break;
      @(negedge vif.clock);
      if (++cycles > 2000) `uvm_fatal("TIMEOUT", $sformatf("command %0h did not complete", rs2))
    end
    verification_pkg::evidence::checks++;
    if (vif.cmd_resp_bits_rob_id !== rob)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0h", vif.cmd_resp_bits_rob_id))
    @(negedge vif.clock);
    for (int port = 0; port < `BB_IN_BW; port++) vif.bank_read_resp_valid[port] = 1'b0;
    for (int port = 0; port < `BB_OUT_BW; port++) vif.bank_write_resp_valid[port] = 1'b0;
  endtask

  // CONFIG (output-stationary, no shift), PRELOAD of B and COMPUTE_PRELOADED of
  // A @ B with A the identity: C must be B's selected group.
  task run_group_select();
    bit [31:0] c[16][16];
    int written = 0;
    issue('h6, `GEMMINI_CONFIG_FUNCT7, 64'd0, 0, 0, c, written);
    issue('h7, `GEMMINI_PRELOAD_FUNCT7, 64'd1 | (64'(GroupB) << 36), 2, 0, c, written);
    issue('h8, `GEMMINI_COMPUTE_PRELOADED_FUNCT7, 64'd2 | (64'(GroupA) << 36) | (64'(GroupB) << 42),
          1, 2, c, written);
    verification_pkg::evidence::checks++;
    if (written != 64) `uvm_fatal("WRITE", $sformatf("%0d row writes, expected 64", written))
    for (int row = 0; row < 16; row++)
      for (int col = 0; col < 16; col++)
        if ($signed(c[row][col]) != (row + 2 * col + 1) % 7 - 3)
          `uvm_fatal("RESULT", $sformatf(
                     "C[%0d][%0d] = %0d, expected %0d",
                     row,
                     col,
                     $signed(
                         c[row][col]
                     ),
                     (row + 2 * col + 1) % 7 - 3
                     ))
    `uvm_info("GEMMINI", "operand group select passed", UVM_LOW)
  endtask

  function void check_no_bank_traffic(string what);
    for (int i = 0; i < `BB_IN_BW; i++)
    if (vif.bank_read_req_valid[i]) `uvm_fatal("BANK", {what, " unexpectedly read a bank"})
    for (int i = 0; i < `BB_OUT_BW; i++)
    if (vif.bank_write_req_valid[i]) `uvm_fatal("BANK", {what, " unexpectedly wrote a bank"})
  endfunction

  task apply_reset();
    vif.cmd_req_valid  = 1'b0;
    vif.cmd_resp_ready = 1'b1;
    for (int i = 0; i < `BB_IN_BW; i++) begin
      vif.bank_read_req_ready[i]  = 1'b1;
      vif.bank_read_resp_valid[i] = 1'b0;
      vif.bank_read_resp_data[i]  = '0;
    end
    for (int i = 0; i < `BB_OUT_BW; i++) begin
      vif.bank_write_req_ready[i]  = 1'b1;
      vif.bank_write_resp_valid[i] = 1'b0;
      vif.bank_write_resp_ok[i]    = 1'b1;
    end
    xif.command_valid  = 1'b0;
    xif.complete_ready = 1'b1;
    xif.write_ready    = 1'b1;
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
