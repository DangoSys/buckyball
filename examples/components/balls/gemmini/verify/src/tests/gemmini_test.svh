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
