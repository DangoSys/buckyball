class fp2int_cmd_driver extends uvm_driver #(fp2int_cmd_item);
  `uvm_component_utils(fp2int_cmd_driver)

  virtual fp2int_if vif;
  uvm_analysis_port #(fp2int_cmd_item) stim_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_ap = new("stim_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fp2int_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("NOVIF", "fp2int_if not found")
    end
  endfunction

  task run_phase(uvm_phase phase);
    fp2int_cmd_item item;
    fp2int_cmd_item stim;

    drive_idle();
    wait (vif.reset === 1'b0);

    forever begin
      seq_item_port.get_next_item(item);
      if (!$cast(stim, item.clone())) begin
        `uvm_fatal("DRV", "failed to clone stimulus item")
      end
      stim_ap.write(stim);
      drive_cmd(item);
      seq_item_port.item_done();
    end
  endtask

  task drive_idle();
    vif.cmd_req_valid <= 1'b0;
    vif.cmd_req_bits_cmd_bid <= '0;
    vif.cmd_req_bits_cmd_funct7 <= '0;
    vif.cmd_req_bits_cmd_iter <= '0;
    vif.cmd_req_bits_cmd_op1_en <= 1'b0;
    vif.cmd_req_bits_cmd_op2_en <= 1'b0;
    vif.cmd_req_bits_cmd_wr_spad_en <= 1'b0;
    vif.cmd_req_bits_cmd_op1_from_spad <= 1'b0;
    vif.cmd_req_bits_cmd_op2_from_spad <= 1'b0;
    vif.cmd_req_bits_cmd_special <= '0;
    vif.cmd_req_bits_cmd_op1_bank <= '0;
    vif.cmd_req_bits_cmd_op2_bank <= '0;
    vif.cmd_req_bits_cmd_wr_bank <= '0;
    vif.cmd_req_bits_cmd_op1_col <= '0;
    vif.cmd_req_bits_cmd_op2_col <= '0;
    vif.cmd_req_bits_cmd_wr_col <= '0;
    vif.cmd_req_bits_cmd_meta_bank <= '0;
    vif.cmd_req_bits_cmd_rs1 <= '0;
    vif.cmd_req_bits_cmd_rs2 <= '0;
    vif.cmd_req_bits_rob_id <= '0;
    vif.cmd_req_bits_is_sub <= 1'b0;
    vif.cmd_req_bits_sub_rob_id <= '0;
  endtask

  task drive_cmd(fp2int_cmd_item item);
    @(posedge vif.clock);
    vif.cmd_req_valid <= 1'b1;
    vif.cmd_req_bits_cmd_bid <= item.bid;
    vif.cmd_req_bits_cmd_funct7 <= item.funct7;
    vif.cmd_req_bits_cmd_iter <= item.iter;
    vif.cmd_req_bits_cmd_op1_en <= item.op1_en;
    vif.cmd_req_bits_cmd_op2_en <= item.op2_en;
    vif.cmd_req_bits_cmd_wr_spad_en <= item.wr_spad_en;
    vif.cmd_req_bits_cmd_op1_from_spad <= item.op1_from_spad;
    vif.cmd_req_bits_cmd_op2_from_spad <= item.op2_from_spad;
    vif.cmd_req_bits_cmd_special <= {32'h00000000, item.scale_bits};
    vif.cmd_req_bits_cmd_op1_bank <= item.op1_bank;
    vif.cmd_req_bits_cmd_op2_bank <= item.op2_bank;
    vif.cmd_req_bits_cmd_wr_bank <= item.wr_bank;
    vif.cmd_req_bits_cmd_op1_col <= item.op1_col;
    vif.cmd_req_bits_cmd_op2_col <= item.op2_col;
    vif.cmd_req_bits_cmd_wr_col <= item.wr_col;
    vif.cmd_req_bits_cmd_meta_bank <= item.meta_bank;
    vif.cmd_req_bits_cmd_rs1 <= item.rs1;
    vif.cmd_req_bits_cmd_rs2 <= item.rs2;
    vif.cmd_req_bits_rob_id <= item.rob_id;
    vif.cmd_req_bits_is_sub <= item.is_sub;
    vif.cmd_req_bits_sub_rob_id <= item.sub_rob_id;

    do begin
      @(posedge vif.clock);
    end while (!vif.cmd_req_ready);

    drive_idle();
  endtask
endclass
