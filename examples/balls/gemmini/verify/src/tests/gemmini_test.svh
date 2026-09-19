class gemmini_ball_test extends uvm_test;
  `uvm_component_utils(gemmini_ball_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
  endfunction

  task run_phase(uvm_phase phase);
    int cycles;
    int unsigned bid;
    phase.raise_objection(this);
    apply_reset();

    @(negedge vif.clock);
    if (!$value$plusargs("BID=%d", bid)) `uvm_fatal("BID", "+BID is required")
    vif.cmd_req_bits_cmd_bid           = bid[4:0];
    vif.cmd_req_bits_cmd_funct7        = `GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNCT7;
    vif.cmd_req_bits_cmd_iter          = '0;
    vif.cmd_req_bits_cmd_op1_en        = 1'b0;
    vif.cmd_req_bits_cmd_op2_en        = 1'b0;
    vif.cmd_req_bits_cmd_wr_spad_en    = 1'b0;
    vif.cmd_req_bits_cmd_op1_from_spad = 1'b0;
    vif.cmd_req_bits_cmd_op2_from_spad = 1'b0;
    vif.cmd_req_bits_cmd_special       = 64'h0003_0002_0001;
    vif.cmd_req_bits_cmd_op1_bank      = '0;
    vif.cmd_req_bits_cmd_op2_bank      = '0;
    vif.cmd_req_bits_cmd_wr_bank       = '0;
    vif.cmd_req_bits_cmd_op1_col       = '0;
    vif.cmd_req_bits_cmd_op2_col       = '0;
    vif.cmd_req_bits_cmd_wr_col        = '0;
    vif.cmd_req_bits_cmd_meta_bank     = '0;
    vif.cmd_req_bits_cmd_rs1           = '0;
    vif.cmd_req_bits_cmd_rs2           = '0;
    vif.cmd_req_bits_rob_id            = 4'h9;
    vif.cmd_req_bits_is_sub            = 1'b1;
    vif.cmd_req_bits_sub_rob_id        = 8'h5a;
    vif.cmd_req_valid                  = 1'b1;

    cycles                             = 0;
    do begin
      @(posedge vif.clock);
      cycles++;
      if (cycles > 20) `uvm_fatal("TIMEOUT", "Gemmini config command was not accepted")
    end while (!vif.cmd_req_ready);
    @(negedge vif.clock);
    vif.cmd_req_valid = 1'b0;

    if (!vif.cmd_resp_valid) `uvm_fatal("RESP", "Gemmini config response was not observed")
    if (vif.cmd_resp_bits_rob_id !== 4'h9 || vif.cmd_resp_bits_is_sub !== 1'b1 ||
        vif.cmd_resp_bits_sub_rob_id !== 8'h5a)
      `uvm_fatal("RESP", $sformatf(
                 "metadata mismatch: rob=%0h is_sub=%0b sub_rob=%0h",
                 vif.cmd_resp_bits_rob_id,
                 vif.cmd_resp_bits_is_sub,
                 vif.cmd_resp_bits_sub_rob_id
                 ))
    for (int i = 0; i < `BB_IN_BW; i++) begin
      if (vif.bank_read_req_valid[i])
        `uvm_fatal("BANK", "Gemmini config command unexpectedly read a bank")
    end
    for (int i = 0; i < `BB_OUT_BW; i++) begin
      if (vif.bank_write_req_valid[i])
        `uvm_fatal("BANK", "Gemmini config command unexpectedly wrote a bank")
    end

    `uvm_info("GEMMINI", "loop-ws config command passed", UVM_LOW)
    phase.drop_objection(this);
  endtask

  task apply_reset();
    vif.cmd_req_valid = 1'b0;
    vif.cmd_resp_ready = 1'b1;
    vif.sub_rob_req_ready = 1'b1;
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
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
  endtask
endclass
